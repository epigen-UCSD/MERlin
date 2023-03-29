import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
import rtree
from typing import List
from cellpose import models as cpmodels
from cellpose import utils
import pandas as pd
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
import pandas
import networkx as nx


class FeatureSavingAnalysisTask(analysistask.AnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName, parallel=True)

    def reset_analysis(self, fragmentIndex: int = None) -> None:
        super().reset_analysis(fragmentIndex)
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

        self.add_dependencies("warp_task", "global_align_task")
        self.set_default_parameters({
            "seed_channel_name": "DAPI",
            "watershed_channel_name": "polyT"
        })

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def run_analysis(self, fragmentIndex):
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
                (watershedOutput == i),
                fragmentIndex,
                self.global_align_task.fov_to_global_transform(fragmentIndex),
                zPos,
            )
            for i in np.unique(watershedOutput)
            if i != 0
        ]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int, filterSigma: float) -> np.ndarray:
        filterSize = int(2 * np.ceil(2 * filterSigma) + 1)
        return np.array(
            [
                cv2.GaussianBlur(
                    self.warp_task.get_aligned_image(fov, channelIndex, z), (filterSize, filterSize), filterSigma
                )
                for z in range(len(self.dataSet.get_z_positions()))
            ]
        )


class CleanCellBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName, parallel=True)

        self.add_dependencies("segment_task", "global_align_task")

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_gpickle("cleaned_cells", self, fragmentIndex)

    def run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.global_align_task.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if fovBoxes[fragmentIndex].intersects(x)])
        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segment_task.get_feature_database().read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segment_task.get_feature_database().read_features(fragmentIndex)
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

        self.add_dependencies("cleaning_task")

    def return_exported_data(self):
        kwargs = {"index_col": 0}
        return self.dataSet.load_dataframe_from_csv("all_cleaned_cells", analysisTask=self.analysis_name, **kwargs)

    def run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaning_task.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, "all_cleaned_cells", analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.add_dependencies("segment_task", "combine_cleaning_task")

    def run_analysis(self, fragmentIndex):
        cleanedCells = self.cleaning_task.return_exported_data()
        originalCells = self.segment_task.get_feature_database().read_features(fragmentIndex)
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

        self.add_dependencies("segment_task")

    def run_analysis(self):
        df = self.segment_task.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, "feature_metadata", self.analysis_name)


class CellposeSegment(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName, parallel=True)

        self.add_dependencies("global_align_task")
        self.add_dependencies("flat_field_task", optional=True)
        self.set_default_parameters({
            "channel": "DAPI",
            "z_pos": None,
            "diameter": None,
            "cellprob_threshold": None,
            "flow_threshold": None,
            "minimum_size": None,
            "dilate_cells": None,
            "downscale_xy": 1,
            "downscale_z": 1
        })

        self.channelIndex = self.dataSet.get_data_organization().get_data_channel_index(self.parameters["channel"])

    def load_mask(self, fragment):
        mask = self.dataSet.load_numpy_analysis_result(
            "mask", self.analysis_name, resultIndex=fragment, subdirectory="masks"
        )
        if mask.ndim == 3:
            shape = (
                mask.shape[0] * self.parameters["downscale_z"],
                mask.shape[1] * self.parameters["downscale_xy"],
                mask.shape[2] * self.parameters["downscale_xy"],
            )
            z_int = np.round(np.linspace(0, mask.shape[0] - 1, shape[0])).astype(int)
            x_int = np.round(np.linspace(0, mask.shape[1] - 1, shape[1])).astype(int)
            y_int = np.round(np.linspace(0, mask.shape[2] - 1, shape[2])).astype(int)
            return mask[z_int][:, x_int][:, :, y_int]
        shape = (
            mask.shape[0] * self.parameters["downscale_xy"],
            mask.shape[1] * self.parameters["downscale_xy"],
        )
        x_int = np.round(np.linspace(0, mask.shape[0] - 1, shape[0])).astype(int)
        y_int = np.round(np.linspace(0, mask.shape[1] - 1, shape[1])).astype(int)
        return mask[x_int][:, y_int]

    def load_metadata(self, fragment):
        return self.dataSet.load_dataframe_from_csv(
            "metadata", self.analysis_name, fragment, subdirectory="metadata"
        )

    def load_image(self, fov, zIndex):
        image = self.dataSet.get_raw_image(self.channelIndex, fov, zIndex)
        if "flat_field_task" in self.dependencies:
            image = self.flat_field_task.process_image(image)
        return image[:: self.parameters["downscale_xy"], :: self.parameters["downscale_xy"]]

    def run_analysis(self, fragment):
        if self.parameters["z_pos"] is not None:
            zIndex = self.dataSet.position_to_z_index(self.parameters["z_pos"])
            inputImage = self.load_image(fragment, zIndex)
        else:
            zPositions = self.dataSet.get_z_positions()[:: self.parameters["downscale_z"]]
            inputImage = np.array(
                [self.load_image(fragment, self.dataSet.position_to_z_index(zIndex)) for zIndex in zPositions]
            )
        model = cpmodels.Cellpose(gpu=False, model_type="cyto2")
        if inputImage.ndim == 2:
            mask, _, _, _ = model.eval(
                inputImage,
                channels=[0, 0],
                diameter=self.parameters["diameter"],
                cellprob_threshold=self.parameters["cellprob_threshold"],
                flow_threshold=self.parameters["flow_threshold"],
            )
        else:
            frames, _, _, _ = model.eval(list(inputImage))
            mask = np.array(utils.stitch3D(frames))
        if self.parameters["minimum_size"]:
            sizes = pd.DataFrame(regionprops_table(mask, properties=["label", "area"]))
            mask[np.isin(mask, sizes[sizes["area"] < self.parameters["minimum_size"]]["label"])] = 0
        if self.parameters["dilate_cells"]:
            if mask.ndim == 2:
                mask = expand_labels(mask, self.parameters["dilate_cells"])
            else:
                mask = np.array([expand_labels(frame, self.parameters["dilate_cells"]) for frame in mask])
        metadata = pd.DataFrame(regionprops_table(mask, properties=["label", "area", "centroid"]))
        columns = ["cell_id", "volume"]
        if mask.ndim == 3:
            columns.append("z")
        columns.extend(["x", "y"])
        metadata.columns = columns
        downscale = self.parameters["downscale_xy"]
        metadata["x"] *= downscale
        metadata["y"] *= downscale
        if mask.ndim == 3:
            metadata["z"] *= self.parameters["downscale_z"]
            metadata["volume"] *= downscale * downscale * self.parameters["downscale_z"]
        else:
            metadata["volume"] *= downscale * downscale
        global_x, global_y = self.global_align_task.fov_coordinates_to_global(
            fragment, metadata[["x", "y"]].T.to_numpy()
        )
        metadata["global_x"] = global_x
        metadata["global_y"] = global_y
        self.dataSet.save_numpy_analysis_result(
            mask, "mask", self.analysis_name, resultIndex=fragment, subdirectory="masks"
        )
        self.dataSet.save_dataframe_to_csv(
            metadata,
            "metadata",
            self.analysis_name,
            resultIndex=fragment,
            subdirectory="metadata",
            index=False,
        )


class LinkCellsInOverlaps(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName, parallel=True)

        self.add_dependencies("segment_task", "global_align_task")

        self.fragment_list = self.dataSet.get_overlap_names()

    def get_links(self, overlapName):
        return self.dataSet.load_pickle_analysis_result(
            "paired_cells", self.analysis_name, resultIndex=overlapName, subdirectory="paired_cells"
        )

    def get_cell_mapping(self):
        try:
            return self.dataSet.load_pickle_analysis_result("cell_links", self.analysis_name)
        except FileNotFoundError:
            linked_sets = []
            for overlapName in self.dataSet.get_overlap_names():
                linked_sets.extend([set(link) for link in self.get_links(overlapName)])
            # Combine sets until they are all disjoint
            # e.g., if there is a (1, 2) and (2, 3) set, combine to (1, 2, 3)
            # This is needed for corners where 4 FOVs overlap
            changed = True
            while changed:
                changed = False
                new = []
                for a in linked_sets:
                    for b in new:
                        if not b.isdisjoint(a):
                            b.update(a)
                            changed = True
                            break
                    else:
                        new.append(a)
                linked_sets = new
            cell_links = {}
            for link_set in linked_sets:
                name = list(link_set)[0]
                for cell in link_set:
                    cell_links[cell] = name
            self.dataSet.save_pickle_analysis_result(cell_links, "cell_links", self.analysis_name)
            return cell_links

    def get_overlap_volumes(self, overlapName):
        return self.dataSet.load_dataframe_from_csv(
            "overlap_volume",
            self.analysis_name,
            resultIndex=overlapName,
            subdirectory="volumes",
        )

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

    def run_analysis(self, fragment: str) -> None:
        """Identify the cells overlapping FOVs that are the same cell."""
        a, b = self.dataSet.get_overlap(fragment)
        pairs = set()
        mask_a = self.segment_task.load_mask(a.fov)
        mask_b = self.segment_task.load_mask(b.fov)
        # Get portions of masks that overlap
        if len(mask_a.shape) == 2:
            strip_a = mask_a[a.xslice, a.yslice]
            strip_b = mask_b[b.xslice, b.yslice]
        elif len(mask_a.shape) == 3:
            strip_a = mask_a[:, a.xslice, a.yslice]
            strip_b = mask_b[:, b.xslice, b.yslice]
        newpairs = self.match_cells_in_overlap(strip_a, strip_b)
        pairs = {(a.fov + "__" + str(x[0]), b.fov + "__" + str(x[1])) for x in newpairs}
        self.dataSet.save_pickle_analysis_result(
            pairs, "paired_cells", self.analysis_name, resultIndex=fragment, subdirectory="paired_cells"
        )
        dfa = pd.DataFrame(regionprops_table(strip_a, properties=["label", "area"]))
        dfa["label"] = a.fov + "__" + dfa["label"].astype(str)
        dfb = pd.DataFrame(regionprops_table(strip_b, properties=["label", "area"]))
        dfb["label"] = b.fov + "__" + dfb["label"].astype(str)
        self.dataSet.save_dataframe_to_csv(
            pd.concat([dfa, dfb]).set_index("label"),
            "overlap_volume",
            self.analysis_name,
            resultIndex=fragment,
            subdirectory="volumes",
        )
