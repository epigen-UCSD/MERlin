from functools import partial
from typing import List

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rtree
from cellpose import models as cpmodels
from cellpose import utils
from scipy import ndimage
from skimage import measure, segmentation
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

from merlin.core import analysistask, dataset
from merlin.util import spatialfeature, watershed


class FeatureSavingAnalysisTask(analysistask.AnalysisTask):
    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def setup(self, *, parallel: bool) -> None:
        super().setup(parallel=parallel)

    def reset_analysis(self, fragmentIndex: int = None) -> None:
        super().reset_analysis()
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

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"warp_task": [], "global_align_task": []})
        self.set_default_parameters({"seed_channel_name": "DAPI", "watershed_channel_name": "polyT"})

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

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"segment_task": [], "global_align_task": []})

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

    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies({"cleaning_task": []})

        self.define_results("all_cleaned_cells")

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

        self.all_cleaned_cells = cleanedCells


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"segment_task": [], "combine_cleaning_task": []})

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

    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies({"segment_task": []})

        self.define_results("feature_metadata")

    def run_analysis(self):
        self.feature_metadata = self.segment_task.get_feature_database().read_feature_metadata()


class CellposeSegment(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"global_align_task": []})
        self.add_dependencies({"flat_field_task": []}, optional=True)

        self.set_default_parameters(
            {
                "channel": "DAPI",
                "z_pos": None,
                "diameter": None,
                "cellprob_threshold": None,
                "flow_threshold": None,
                "minimum_size": None,
                "dilate_cells": None,
                "downscale_xy": 1,
                "downscale_z": 1,
            }
        )

        self.define_results("mask", ("cell_metadata", {"index": False}))

        self.channelIndex = self.dataSet.get_data_organization().get_data_channel_index(self.parameters["channel"])

    def load_mask(self):
        mask = self.load_result("mask")
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

    def load_cell_metadata(self):
        return self.dataSet.load_dataframe_from_csv(
            "cell_metadata", self.analysis_name, self.fragment, subdirectory="cell_metadata"
        )

    def load_image(self, zIndex):
        image = self.dataSet.get_raw_image(self.channelIndex, self.fragment, zIndex)
        if "flat_field_task" in self.dependencies:
            image = self.flat_field_task.process_image(image)
        return image[:: self.parameters["downscale_xy"], :: self.parameters["downscale_xy"]]

    def run_analysis(self):
        if self.parameters["z_pos"] is not None:
            zIndex = self.dataSet.position_to_z_index(self.parameters["z_pos"])
            inputImage = self.load_image(zIndex)
        else:
            zPositions = self.dataSet.get_z_positions()[:: self.parameters["downscale_z"]]
            inputImage = np.array([self.load_image(self.dataSet.position_to_z_index(zIndex)) for zIndex in zPositions])
        model = cpmodels.Cellpose(gpu=False, model_type="cyto2")
        segment = partial(
            model.eval,
            channels=[0, 0],
            diameter=self.parameters["diameter"],
            cellprob_threshold=self.parameters["cellprob_threshold"],
            flow_threshold=self.parameters["flow_threshold"],
        )
        if inputImage.ndim == 2:
            mask, _, _, _ = segment(inputImage)
        else:
            frames, _, _, _ = segment(list(inputImage))
            mask = np.array(utils.stitch3D(frames))
        if self.parameters["minimum_size"]:
            sizes = pd.DataFrame(regionprops_table(mask, properties=["label", "area"]))
            mask[np.isin(mask, sizes[sizes["area"] < self.parameters["minimum_size"]]["label"])] = 0
        if self.parameters["dilate_cells"]:
            if mask.ndim == 2:
                mask = expand_labels(mask, self.parameters["dilate_cells"])
            else:
                mask = np.array([expand_labels(frame, self.parameters["dilate_cells"]) for frame in mask])
        cell_metadata = pd.DataFrame(regionprops_table(mask, properties=["label", "area", "centroid"]))
        columns = ["cell_id", "volume"]
        if mask.ndim == 3:
            columns.append("z")
        columns.extend(["x", "y"])
        cell_metadata.columns = columns
        downscale = self.parameters["downscale_xy"]
        cell_metadata["x"] *= downscale
        cell_metadata["y"] *= downscale
        if mask.ndim == 3:
            cell_metadata["z"] *= self.parameters["downscale_z"]
            cell_metadata["volume"] *= downscale * downscale * self.parameters["downscale_z"]
        else:
            cell_metadata["volume"] *= downscale * downscale
        global_x, global_y = self.global_align_task.fov_coordinates_to_global(
            self.fragment, cell_metadata[["x", "y"]].T.to_numpy()
        )
        cell_metadata["global_x"] = global_x
        cell_metadata["global_y"] = global_y
        self.mask = mask
        self.cell_metadata = cell_metadata

    def metadata(self) -> dict:
        return {"cells": len(self.cell_metadata)}


class CellposeSegment3D(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True, threads=8)

        self.add_dependencies({"global_align_task": []})
        self.add_dependencies({"flat_field_task": []}, optional=True)

        self.set_default_parameters(
            {
                "channel": "DAPI",
                "cellpose_model": "nuclei",
                "diameter": None,
                "cellprob_threshold": None,
                "flow_threshold": None,
                "minimum_size": None,
                "downscale_xy": 1,
                "downscale_z": 1,
            }
        )

        self.define_results("mask", ("cell_metadata", {"index": False}))

        self.channelIndex = self.dataSet.get_data_organization().get_data_channel_index(self.parameters["channel"])

    def load_mask(self):
        mask = self.load_result("mask")
        shape = (
            mask.shape[0] * self.parameters["downscale_z"],
            mask.shape[1] * self.parameters["downscale_xy"],
            mask.shape[2] * self.parameters["downscale_xy"],
        )
        z_int = np.round(np.linspace(0, mask.shape[0] - 1, shape[0])).astype(int)
        x_int = np.round(np.linspace(0, mask.shape[1] - 1, shape[1])).astype(int)
        y_int = np.round(np.linspace(0, mask.shape[2] - 1, shape[2])).astype(int)
        return mask[z_int][:, x_int][:, :, y_int]

    def load_cell_metadata(self):
        return self.dataSet.load_dataframe_from_csv(
            "cell_metadata", self.analysis_name, self.fragment, subdirectory="cell_metadata"
        )

    def load_image(self, zIndex):
        image = self.dataSet.get_raw_image(self.channelIndex, self.fragment, zIndex)
        if "flat_field_task" in self.dependencies:
            image = self.flat_field_task.process_image(
                image, self.dataSet.get_data_organization().get_data_channel_color(self.channelIndex)
            )
        return image[:: self.parameters["downscale_xy"], :: self.parameters["downscale_xy"]]

    def slice_pair_to_info(self, pair):
        sl1, sl2 = pair
        xm, ym, sx, sy = sl2.start, sl1.start, sl2.stop - sl2.start, sl1.stop - sl1.start
        A = sx * sy
        return [xm, ym, sx, sy, A]

    def get_coords(self, imlab1, infos1, cell1):
        xm, ym, sx, sy, A, icl = infos1[cell1 - 1]
        return np.array(np.where(imlab1[ym : ym + sy, xm : xm + sx] == icl)).T + [ym, xm]

    def cells_to_coords(self, imlab1, return_labs=False):
        """return the coordinates of cells with some additional info"""
        infos1 = [
            self.slice_pair_to_info(pair) + [icell + 1]
            for icell, pair in enumerate(ndimage.find_objects(imlab1))
            if pair is not None
        ]
        cms1 = np.array([np.mean(self.get_coords(imlab1, infos1, cl + 1), 0) for cl in range(len(infos1))])
        cms1 = cms1[:, ::-1]
        ies = [info[-1] for info in infos1]
        if return_labs:
            return imlab1.copy(), infos1, cms1, ies
        return imlab1.copy(), infos1, cms1

    def resplit(self, cells1, cells2, nmin=100):
        """intermediate function used by standard_segmentation.
        Decide when comparing two planes which cells to split"""
        imlab1, infos1, cms1 = self.cells_to_coords(cells1)
        imlab2, infos2, cms2 = self.cells_to_coords(cells2)

        # find centers 2 within the cells1 and split cells1
        cms2_ = np.round(cms2).astype(int)
        cells2_1 = imlab1[cms2_[:, 1], cms2_[:, 0]]
        imlab1_cells = [0] + [info[-1] for info in infos1]
        cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]  # reorder coords
        # [for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
        dic_cell2_1 = {}
        for cell1, cell2 in enumerate(cells2_1):
            dic_cell2_1[cell2] = dic_cell2_1.get(cell2, []) + [cell1 + 1]
        dic_cell2_1_split = {cell: dic_cell2_1[cell] for cell in dic_cell2_1 if len(dic_cell2_1[cell]) > 1 and cell > 0}
        cells1_split = list(dic_cell2_1_split.keys())
        imlab1_cp = imlab1.copy()
        number_of_cells_to_split = len(cells1_split)
        for cell1_split in cells1_split:
            count = np.max(imlab1_cp) + 1
            cells2_to1 = dic_cell2_1_split[cell1_split]
            X1 = self.get_coords(imlab1, infos1, cell1_split)
            X2s = [self.get_coords(imlab2, infos2, cell2) for cell2 in cells2_to1]
            from scipy.spatial.distance import cdist

            X1_K = np.argmin([np.min(cdist(X1, X2), axis=-1) for X2 in X2s], 0)

            for k in range(len(X2s)):
                X_ = X1[X1_K == k]
                if len(X_) > nmin:
                    imlab1_cp[X_[:, 0], X_[:, 1]] = count + k
                else:
                    # number_of_cells_to_split-=1
                    pass
        imlab1_, infos1_, cms1_ = self.cells_to_coords(imlab1_cp)
        return imlab1_, infos1_, cms1_, number_of_cells_to_split

    def converge(self, cells1, cells2):
        imlab1, infos1, cms1, labs1 = self.cells_to_coords(cells1, return_labs=True)
        imlab2, infos2, cms2 = self.cells_to_coords(cells2)

        # find centers 2 within the cells1 and split cells1
        cms2_ = np.round(cms2).astype(int)
        cells2_1 = imlab1[cms2_[:, 1], cms2_[:, 0]]
        imlab1_cells = [0] + [info[-1] for info in infos1]
        cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]  # reorder coords
        # [for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
        dic_cell2_1 = {}
        for cell1, cell2 in enumerate(cells2_1):
            dic_cell2_1[cell2] = dic_cell2_1.get(cell2, []) + [cell1 + 1]

        dic_cell2_1_match = {cell: dic_cell2_1[cell] for cell in dic_cell2_1 if cell > 0}
        cells2_kp = [e_ for e in dic_cell2_1_match for e_ in dic_cell2_1_match[e]]
        modify_cells2 = np.setdiff1d(np.arange(len(cms2)), cells2_kp)
        imlab2_ = imlab2 * 0
        for cell1 in dic_cell2_1_match:
            for cell2 in dic_cell2_1_match[cell1]:
                xm, ym, sx, sy, A, icl = infos2[cell2 - 1]
                im_sm = imlab2[ym : ym + sy, xm : xm + sx]
                imlab2_[ym : ym + sy, xm : xm + sx][im_sm == icl] = labs1[cell1 - 1]
        count_cell = max(np.max(imlab2_), np.max(labs1))
        for cell2 in modify_cells2:
            count_cell += 1
            xm, ym, sx, sy, A, icl = infos2[cell2 - 1]
            im_sm = imlab2[ym : ym + sy, xm : xm + sx]
            imlab2_[ym : ym + sy, xm : xm + sx][im_sm == icl] = count_cell
        return imlab1, imlab2_

    def run_analysis(self):
        model = cpmodels.Cellpose(gpu=False, model_type=self.parameters["cellpose_model"])
        zPositions = self.dataSet.get_z_positions()[:: self.parameters["downscale_z"]]
        im_dapi_3d = np.array([self.load_image(self.dataSet.position_to_z_index(zIndex)) for zIndex in zPositions])
        masks_all = []
        for img in im_dapi_3d:
            masks, _, _, _ = model.eval(
                img,
                diameter=self.parameters["diameter"],
                channels=[0, 0],
                flow_threshold=self.parameters["flow_threshold"],
                cellprob_threshold=self.parameters["cellprob_threshold"],
                min_size=50,
            )
            masks_all.append(masks)
        masks_all = np.array(masks_all)

        sec_half = list(np.arange(int(len(masks_all) / 2), len(masks_all) - 1))
        first_half = list(np.arange(0, int(len(masks_all) / 2)))[::-1]
        indexes = first_half + sec_half
        masks_all_cp = masks_all.copy()
        max_split = 1
        niter = 0
        while max_split > 0 and niter < 3:
            max_split = 0
            for index in indexes:
                cells1, cells2 = masks_all_cp[index], masks_all_cp[index + 1]
                if not cells1.any() or not cells2.any():
                    continue
                imlab1_, infos1_, cms1_, no1 = self.resplit(cells1, cells2)
                imlab2_, infos2_, cms2_, no2 = self.resplit(cells2, cells1)
                masks_all_cp[index], masks_all_cp[index + 1] = imlab1_, imlab2_
                max_split += max(no1, no2)
                # print(no1,no2)
            niter += 1
        masks_all_cpf = masks_all_cp.copy()
        for index in range(len(masks_all_cpf) - 1):
            cells1, cells2 = masks_all_cpf[index], masks_all_cpf[index + 1]
            if not cells1.any() or not cells2.any():
                continue
            cells1_, cells2_ = self.converge(cells1, cells2)
            masks_all_cpf[index + 1] = cells2_
        self.mask = masks_all_cpf
        if self.parameters["minimum_size"]:
            sizes = pd.DataFrame(regionprops_table(self.mask, properties=["label", "area"]))
            self.mask[np.isin(self.mask, sizes[sizes["area"] < self.parameters["minimum_size"]]["label"])] = 0

        cell_metadata = pd.DataFrame(regionprops_table(self.mask, properties=["label", "area", "centroid"]))
        cell_metadata.columns = ["cell_id", "volume", "z", "x", "y"]
        downscale = self.parameters["downscale_xy"]
        cell_metadata["x"] *= downscale
        cell_metadata["y"] *= downscale
        cell_metadata["z"] *= self.parameters["downscale_z"]
        cell_metadata["volume"] *= downscale * downscale * self.parameters["downscale_z"]
        global_x, global_y = self.global_align_task.fov_coordinates_to_global(
            self.fragment, cell_metadata[["x", "y"]].T.to_numpy()
        )
        cell_metadata["global_x"] = global_x
        cell_metadata["global_y"] = global_y
        self.cell_metadata = cell_metadata


class LinkCellsInOverlaps(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"segment_task": [], "global_align_task": []})

        self.define_results("overlap_volume", "paired_cells")
        self.define_results("cell_mapping", ("cell_metadata", {"index": True}), final=True)

        self.fragment_list = self.dataSet.get_overlap_names()

    def get_links(self, overlapName):
        return self.dataSet.load_pickle_analysis_result(
            "paired_cells", self.analysis_name, resultIndex=overlapName, subdirectory="paired_cells"
        )

    def get_cell_mapping(self):
        try:
            return self.dataSet.load_pickle_analysis_result("cell_mapping", self.analysis_name)
        except FileNotFoundError:
            linked_sets = []
            for overlap_name in self.dataSet.get_overlap_names():
                linked_sets.extend([set(link) for link in self.get_links(overlap_name)])
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
            return cell_links

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

    def run_analysis(self) -> None:
        """Identify the cells overlapping FOVs that are the same cell."""
        a, b = self.dataSet.get_overlap(self.fragment)
        pairs = set()
        segment_a = self.dataSet.load_analysis_task(self.parameters["segment_task"], a.fov)
        segment_b = self.dataSet.load_analysis_task(self.parameters["segment_task"], b.fov)
        mask_a = segment_a.load_mask()
        mask_b = segment_b.load_mask()
        # Get portions of masks that overlap
        if len(mask_a.shape) == 2:
            strip_a = mask_a[a.xslice, a.yslice]
            strip_b = mask_b[b.xslice, b.yslice]
        elif len(mask_a.shape) == 3:
            strip_a = mask_a[:, a.xslice, a.yslice]
            strip_b = mask_b[:, b.xslice, b.yslice]
        newpairs = self.match_cells_in_overlap(strip_a, strip_b)
        pairs = {(a.fov + "__" + str(x[0]), b.fov + "__" + str(x[1])) for x in newpairs}
        self.paired_cells = pairs

        dfa = pd.DataFrame(regionprops_table(strip_a, properties=["label", "area"]))
        dfa["label"] = a.fov + "__" + dfa["label"].astype(str)
        dfb = pd.DataFrame(regionprops_table(strip_b, properties=["label", "area"]))
        dfb["label"] = b.fov + "__" + dfb["label"].astype(str)
        self.overlap_volume = pd.concat([dfa, dfb]).set_index("label")

    def combine_overlap_volumes(self):
        volumes = []
        for fragment in self.fragment_list:
            self.fragment = fragment
            volumes.append(self.load_result("overlap_volume"))
        self.fragment = ""
        volumes = pd.concat(volumes)
        return volumes.groupby("label").max()

    def finalize_analysis(self):
        dfs = []
        self.cell_mapping = self.get_cell_mapping()
        for fov in self.dataSet.get_fovs():
            self.segment_task.fragment = fov
            cell_metadata = self.segment_task.load_cell_metadata()
            cell_metadata["cell_id"] = fov + "__" + cell_metadata["cell_id"].astype(str)
            cell_metadata = cell_metadata.rename(columns={"volume": "fov_volume"})
            dfs.append(cell_metadata)
        cell_metadata = pd.concat(dfs).set_index("cell_id")
        cell_metadata["overlap_volume"] = self.combine_overlap_volumes()
        cell_metadata["overlap_volume"] = cell_metadata["overlap_volume"].fillna(0)
        cell_metadata["nonoverlap_volume"] = cell_metadata["fov_volume"] - cell_metadata["overlap_volume"]
        cell_metadata.index = [self.cell_mapping.get(cell_id, cell_id) for cell_id in cell_metadata.index]
        cell_metadata.index.name = "cell_id"
        cell_metadata = cell_metadata.groupby("cell_id").agg(
            {"global_x": "mean", "global_y": "mean", "overlap_volume": "mean", "nonoverlap_volume": "sum"}
        )
        cell_metadata["volume"] = cell_metadata["overlap_volume"] + cell_metadata["nonoverlap_volume"]
        cell_metadata = cell_metadata.drop(columns=["overlap_volume", "nonoverlap_volume"])
        self.cell_metadata = cell_metadata
