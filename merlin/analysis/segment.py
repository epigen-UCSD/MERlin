import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
import rtree
from shapely import geometry
from typing import List, Dict
from scipy.spatial import cKDTree
from cellpose import models as cpmodels
import pandas as pd
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
import pandas
import networkx as nx


class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)


class WatershedSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.

    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if "seed_channel_name" not in self.parameters:
            self.parameters["seed_channel_name"] = "DAPI"
        if "watershed_channel_name" not in self.parameters:
            self.parameters["watershed_channel_name"] = "polyT"

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters["warp_task"], self.parameters["global_align_task"]]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(self.parameters["global_align_task"])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(self.parameters["seed_channel_name"])
        seedImages = self._read_and_filter_image_stack(fragmentIndex, seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters["watershed_channel_name"]
        )
        watershedImages = self._read_and_filter_image_stack(fragmentIndex, watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(watershed.extract_seeds(seedImages))
        normalizedWatershed, watershedMask = watershed.prepare_watershed_images(watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed,
            measure.label(seeds),
            mask=watershedMask,
            connectivity=np.ones((3, 3, 3)),
            watershed_line=True,
        )

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [
            spatialfeature.SpatialFeature.feature_from_label_matrix(
                (watershedOutput == i), fragmentIndex, globalTask.fov_to_global_transform(fragmentIndex), zPos
            )
            for i in np.unique(watershedOutput)
            if i != 0
        ]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int, filterSigma: float) -> np.ndarray:
        filterSize = int(2 * np.ceil(2 * filterSigma) + 1)
        warpTask = self.dataSet.load_analysis_task(self.parameters["warp_task"])
        return np.array(
            [
                cv2.GaussianBlur(
                    warpTask.get_aligned_image(fov, channelIndex, z), (filterSize, filterSize), filterSigma
                )
                for z in range(len(self.dataSet.get_z_positions()))
            ]
        )


class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(self.parameters["segment_task"])
        self.alignTask = self.dataSet.load_analysis_task(self.parameters["global_align_task"])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters["segment_task"], self.parameters["global_align_task"]]

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_gpickle("cleaned_cells", self, fragmentIndex)

    def _run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if fovBoxes[fragmentIndex].intersects(x)])
        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segmentTask.get_feature_database().read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database().read_features(fragmentIndex)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(graph, cells, spatialTree, fragmentIndex, allFOVs, fovBoxes)

        self.dataSet.save_graph_as_gpickle(graph, "cleaned_cells", self, fragmentIndex)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(self.parameters["cleaning_task"])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters["cleaning_task"]]

    def return_exported_data(self):
        kwargs = {"index_col": 0}
        return self.dataSet.load_dataframe_from_csv("all_cleaned_cells", analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, "all_cleaned_cells", analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(self.parameters["segment_task"])
        self.cleaningTask = self.dataSet.load_analysis_task(self.parameters["combine_cleaning_task"])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters["segment_task"], self.parameters["combine_cleaning_task"]]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database().read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells["originalFOV"] == fragmentIndex]
        cleanedGroups = cleanedC.groupby("assignedFOV")
        for k, g in cleanedGroups:
            cellsToConsider = g["cell_id"].values.tolist()
            featureList = [x for x in originalCells if str(x.get_feature_id()) in cellsToConsider]
            featureDB.write_features(featureList, fragmentIndex)


class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(self.parameters["segment_task"])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters["segment_task"]]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, "feature_metadata", self.analysisName)


class CellposeSegment(analysistask.ParallelAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if "channel" not in self.parameters:
            self.parameters["channel"] = "DAPI"
        if "diameter" not in self.parameters:
            self.parameters["diameter"] = None
        if "cellprob_threshold" not in self.parameters:
            self.parameters["cellprob_threshold"] = None
        if "flow_threshold" not in self.parameters:
            self.parameters["flow_threshold"] = None
        if "minimum_size" not in self.parameters:
            self.parameters["minimum_size"] = None
        if "dilate_cells" not in self.parameters:
            self.parameters["dilate_cells"] = None

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _run_analysis(self, fragmentIndex):
        # Only 2D segmentation for now. Use middle z-slice.
        zPositions = self.dataSet.get_z_positions()
        zIndex = int(len(zPositions) // 2)
        channelIndex = self.dataSet.get_data_organization().get_data_channel_index(self.parameters["channel"])
        inputImage = self.dataSet.get_raw_image(channelIndex, fragmentIndex, zIndex)
        model = cpmodels.Cellpose(gpu=False, model_type="cyto2")
        mask, _, _, _ = model.eval(
            inputImage,
            channels=[0, 0],
            diameter=self.parameters["diameter"],
            cellprob_threshold=self.parameters["cellprob_threshold"],
            flow_threshold=self.parameters["flow_threshold"],
        )
        if self.parameters["minimum_size"]:
            sizes = pd.DataFrame(regionprops_table(mask, properties=["label", "area"]))
            mask[np.isin(mask, sizes[sizes["area"] < self.parameters["minimum_size"]]["label"])] = 0
        if self.parameters["dilate_cells"]:
            mask = expand_labels(mask, self.parameters["dilate_cells"])
        metadata = pd.DataFrame(regionprops_table(mask, properties=["label", "area", "centroid"]))
        metadata.columns = ["cell_id", "volume", "x", "y"]
        self.dataSet.save_numpy_analysis_result(
            mask, "mask", self.get_analysis_name(), resultIndex=fragmentIndex, subdirectory="masks"
        )
        self.dataSet.save_dataframe_to_csv(
            metadata,
            "metadata",
            self.get_analysis_name(),
            resultIndex=fragmentIndex,
            subdirectory="metadata",
            index=False,
        )


class LinkCellsInOverlaps(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters["segment_task"], self.parameters["global_align_task"]]

    def match_cells_in_overlap(self, strip_a: np.ndarray, strip_b: np.ndarray):
        """Find cells in overlapping regions of two FOVs that are the same cells.
        :param strip_a: The overlapping region of the segmentation mask from one FOV.
        :param strip_b: The overlapping region of the segmentation mask from another FOV.
        :return: A set of pairs of ints (tuples) representing the mask labels from each mask
            that are the same cell. For example, the tuple `(23, 45)` means mask label 23 from
            the mask given by `strip_a` is the same cell as mask label 45 in the mask given by
            `strip_b`.
        """
        # Pair up pixels in overlap regions
        # This could be more precise by drift correcting between the two FOVs
        p = np.array([strip_a.flatten(), strip_b.flatten()]).T
        # Remove pixel pairs with 0s (no cell) and count overlapping areas between cells
        ps, c = np.unique(p[np.all(p != 0, axis=1)], axis=0, return_counts=True)
        # For each cell from A, find the cell in B it overlaps with most (s1)
        # Do the same from B to A (s2)
        df = pd.DataFrame(np.hstack((ps, np.array([c]).T)), columns=["a", "b", "count"])
        s1 = {
            tuple(x)
            for x in df.sort_values(["a", "count"], ascending=[True, False])
            .groupby("a")
            .first()
            .reset_index()[["a", "b"]]
            .values.tolist()
        }
        s2 = {
            tuple(x)
            for x in df.sort_values(["b", "count"], ascending=[True, False])
            .groupby("b")
            .first()
            .reset_index()[["a", "b"]]
            .values.tolist()
        }
        # Only keep the pairs found in both directions
        return s1 & s2

    def _run_analysis(self):
        """Identify the cells overlapping FOVs that are the same cell."""
        globalTask = self.dataSet.load_analysis_task(self.parameters["global_align_task"])
        pairs = set()
        for a, b in globalTask.find_fov_overlaps():
            # Get portions of masks that overlap
            if len(self[a.fov].shape) == 2:
                strip_a = self[a.fov][a.xslice, a.yslice]
                strip_b = self[b.fov][b.xslice, b.yslice]
            elif len(self[a.fov].shape) == 3:
                strip_a = self[a.fov][:, a.xslice, a.yslice]
                strip_b = self[b.fov][:, b.xslice, b.yslice]
            newpairs = self.match_cells_in_overlap(strip_a, strip_b)
            pairs.update({(a.fov * 10000 + x[0], b.fov * 10000 + x[1]) for x in newpairs})
        linked_sets = [set([a, b]) for a, b in pairs]
        # Combine sets until they are all disjoint
        # e.g., if there is a (1, 2) and (2, 3) set, combine to (1, 2, 3)
        # This is needed for corners where 4 FOVs overlap
        changed = True
        while changed:
            changed = False
            new: List[set] = []
            for a in linked_sets:
                for b in new:
                    if not b.isdisjoint(a):
                        b.update(a)
                        changed = True
                        break
                else:
                    new.append(a)
            linked_sets = new
        self.dataSet.save_pickle_analysis_result(linked_sets, "cell_links", self.get_analysis_name())


class CombineCellposeMetadata(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(self.parameters["segment_task"])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters["segment_task"]]

    def _run_analysis(self):
        dfs = []
        for fov in self.dataSet.get_fovs():
            df = self.dataSet.load_dataframe_from_csv("metadata", self.parameters["segment_task"], fov, "metadata")
            df["cell_id"] += fov * 10000
            dfs.append(df)
        metadata = pd.concat(dfs)
        metadata = metadata.set_index("cell_id")

        self.dataSet.save_dataframe_to_csv(metadata, "cell_metadata", self.get_analysis_name())
