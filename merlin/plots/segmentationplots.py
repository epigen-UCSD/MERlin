import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import regionprops_table

from merlin.analysis.partition import PartitionBarcodesFromMask
from merlin.analysis.segment import CellposeSegment, FeatureSavingAnalysisTask, LinkCellsInOverlaps
from merlin.plots import tools
from merlin.plots._base import AbstractPlot


class SegmentationBoundaryPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"segment_task": FeatureSavingAnalysisTask})

    def create_plot(self, **kwargs):
        feature_db = kwargs["tasks"]["segment_task"].get_feature_database()
        features = feature_db.read_features()

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", "datalim")

        if len(features) == 0:
            return fig

        z_position = 0
        if len(features[0].get_boundaries()) > 1:
            z_position = int(len(features[0].get_boundaries()) / 2)

        features_z = [feature.get_boundaries()[int(z_position)] for feature in features]
        features_z = [x for y in features_z for x in y]
        coords = [
            [feature.exterior.coords.xy[0].tolist(), feature.exterior.coords.xy[1].tolist()] for feature in features_z
        ]
        coords = [x for y in coords for x in y]
        plt.plot(*coords)

        plt.xlabel("X position (microns)")
        plt.ylabel("Y position (microns)")
        plt.title("Segmentation boundaries")
        return fig


class CellposeBoundaryPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"segment_task": "all", "link_cell_task": LinkCellsInOverlaps})
        self.formats = [".png"]

    def plot_mask(self, fov, ax) -> None:
        self.segment_task.fragment = fov
        mask = self.segment_task.load_mask()
        channel = self.dataset.get_data_organization().get_data_channel_index(self.segment_task.parameters["channel"])
        if "z_pos" in self.segment_task.parameters and self.segment_task.parameters["z_pos"] is not None:
            z_index = self.dataset.position_to_z_index(self.segment_task.parameters["z_pos"])
            image = self.dataset.get_raw_image(channel, fov, z_index)
        else:
            z_positions = self.dataset.get_z_positions()
            z_index = z_positions[len(z_positions) // 2]
            image = self.dataset.get_raw_image(channel, fov, z_index)
            mask = mask[int(z_index)]

        ax.imshow(image, cmap="gray")
        ax.contour(
            mask,
            [x + 0.5 for x in np.unique(mask)],
            colors="tab:blue",
            linewidths=1,
            zorder=2,
        )
        ax.contourf(mask, [x + 0.5 for x in np.unique(mask)], colors="tab:blue", alpha=0.2)
        ax.axis("off")

    def create_plot(self, **kwargs) -> plt.Figure:
        self.segment_task = kwargs["tasks"]["segment_task"]
        self.dataset = self.segment_task.dataSet
        link_cell_task = kwargs["tasks"]["link_cell_task"]
        metadata = link_cell_task.load_result("cell_metadata")
        metadata["fov"] = [cell_id.split("__")[0] for cell_id in metadata.index]

        fig, ax = plt.subplots(3, 2, figsize=(8, 12), dpi=300)
        counts = metadata.groupby("fov").count().sort_values("volume")
        fovs = counts.index
        self.plot_mask(fovs[len(fovs) // 2], ax[0, 0])
        ax[0, 0].set_title(f"FOV {fovs[len(fovs)//2]} - {counts.loc[fovs[len(fovs)//2]].volume} cells")
        self.plot_mask(fovs[len(fovs) // 2 - 1], ax[0, 1])
        ax[0, 1].set_title(f"FOV {fovs[len(fovs)//2-1]} - {counts.loc[fovs[len(fovs)//2-1]].volume} cells")
        self.plot_mask(fovs[0], ax[1, 0])
        ax[1, 0].set_title(f"FOV {fovs[0]} - {counts.loc[fovs[0]].volume} cells")
        self.plot_mask(fovs[1], ax[1, 1])
        ax[1, 1].set_title(f"FOV {fovs[1]} - {counts.loc[fovs[1]].volume} cells")
        self.plot_mask(fovs[-1], ax[2, 0])
        ax[2, 0].set_title(f"FOV {fovs[-1]} - {counts.loc[fovs[-1]].volume} cells")
        self.plot_mask(fovs[-2], ax[2, 1])
        ax[2, 1].set_title(f"FOV {fovs[-2]} - {counts.loc[fovs[-2]].volume} cells")

        return fig


class CellVolumeHistogramPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"link_cell_task": LinkCellsInOverlaps})

    def create_plot(self, **kwargs) -> plt.Figure:
        link_cell_task = kwargs["tasks"]["link_cell_task"]
        metadata = link_cell_task.load_result("cell_metadata")
        fig = tools.plot_histogram(metadata, "volume")
        plt.xlabel("Cell volume (pixels)")
        return fig


class BarcodesAssignedToCellsPlot(AbstractPlot):
    def __init__(self, plot_task) -> None:
        super().__init__(plot_task)
        self.set_required_tasks({"partition_task": PartitionBarcodesFromMask, "segment_task": "all"})
        self.formats = [".png"]

    def create_plot(self, **kwargs) -> plt.Figure:
        partition_task = kwargs["tasks"]["partition_task"]
        segment_task = kwargs["tasks"]["segment_task"]
        fov = self.plot_task.dataSet.get_fovs()[42]
        partition_task.fragment = fov
        segment_task.fragment = fov
        segment_task.parameters["downscale_xy"] = 1
        barcodes = partition_task.load_result("barcodes")
        image = segment_task.load_image(zIndex=10)
        incells = barcodes[barcodes["cell_id"] != f"{fov}__0"]
        outcells = barcodes[barcodes["cell_id"] == f"{fov}__0"]
        fig = plt.figure(dpi=200, figsize=(10, 10))
        plt.imshow(image, cmap="gray", vmax=np.percentile(image, 99))
        plt.scatter(incells["y"], incells["x"], s=1, alpha=0.5, c="tab:blue", marker=".")
        plt.scatter(outcells["y"], outcells["x"], s=1, alpha=0.5, c="tab:red", marker=".")
        plt.axis("off")
        return fig


class LinkCellsPlot(AbstractPlot):
    def __init__(self, plot_task) -> None:
        super().__init__(plot_task)
        self.set_required_tasks({"link_cell_task": LinkCellsInOverlaps})
        self.formats = [".png"]

    def create_plot(self, **kwargs) -> plt.Figure:
        link_task = kwargs["tasks"]["link_cell_task"]

        # Find a FOV with a reasonable number of links to show
        for link_fragment in link_task.fragment_list:
            links = link_task.get_links(link_fragment)
            fragment1, fragment2 = link_fragment.split("__")
            if len(links) > 10:
                break

        # Find the orientation of the two FOVs (vertical/horizontal)
        diffs = (
            link_task.dataSet.get_stage_positions().loc[fragment1]
            - link_task.dataSet.get_stage_positions().loc[fragment2]
        )
        axis = np.abs(diffs).argmax()
        if diffs[axis] > 0:
            fragment1, fragment2 = fragment2, fragment1

        # Load the masks
        segtask1 = link_task.dataSet.load_analysis_task(self.plot_task.parameters["segment_task"], fragment=fragment1)
        mask1 = segtask1.load_result("mask")
        segtask2 = link_task.dataSet.load_analysis_task(self.plot_task.parameters["segment_task"], fragment=fragment2)
        mask2 = segtask2.load_result("mask")

        # Load the background images
        if "z_pos" in segtask1.parameters and segtask1.parameters["z_pos"] is not None:
            z_index = segtask1.dataSet.position_to_z_index(segtask1.parameters["z_pos"])
        else:
            z_index = len(mask1)//2
            mask1 = mask1[z_index]
        img1 = segtask1.load_image(z_index)

        if "z_pos" in segtask2.parameters and segtask2.parameters["z_pos"] is not None:
            z_index = segtask2.dataSet.position_to_z_index(segtask2.parameters["z_pos"])
        else:
            z_index = len(mask2)//2
            mask2 = mask2[z_index]
        img2 = segtask2.load_image(z_index)

        # Combine the two FOVs into one
        if axis == 1:
            segimg = np.zeros((img1.shape[0], img1.shape[0] * 2 + 100))
            segimg[:, : img1.shape[0]] = img1
            segimg[:, img1.shape[0] + 100 :] = img2
            mask = np.zeros_like(segimg)
            mask[:, : img1.shape[0]] = mask1
            mask[:, img1.shape[0] + 100 :] = mask2
        else:
            segimg = np.zeros((img1.shape[0] * 2 + 100, img1.shape[0]))
            segimg[: img1.shape[0], :] = img1
            segimg[img1.shape[0] + 100 :, :] = img2
            mask = np.zeros_like(segimg)
            mask[: img1.shape[0], :] = mask1
            mask[img1.shape[0] + 100 :, :] = mask2

        # Get the cell centroids
        props1 = pd.DataFrame(regionprops_table(mask1, properties=["label", "centroid"]))
        props1 = props1.set_index("label")
        props2 = pd.DataFrame(regionprops_table(mask2, properties=["label", "centroid"]))
        props2 = props2.set_index("label")
        props2[f"centroid-{axis}"] = props2[f"centroid-{axis}"] + img1.shape[0] + 100

        # Plot everything
        figsize = (6, 12) if axis == 0 else (12, 6)
        fig = plt.figure(figsize=figsize, dpi=150)
        plt.imshow(segimg, cmap="gray", vmax=np.percentile(segimg, 99))
        plt.contour(mask, [0.5 + x for x in np.unique(mask)], colors="tab:blue", linewidths=1)
        for cell1, cell2 in links:
            if diffs[axis] > 0:
                cell1, cell2 = cell2, cell1
            id1 = int(cell1.split("__")[1])
            id2 = int(cell2.split("__")[1])
            if id1 in props1.index and id2 in props2.index:
                y1, x1 = props1.loc[id1]
                y2, x2 = props2.loc[id2]
                plt.plot([x1, x2], [y1, y2], c="tab:red", alpha=0.5)
        plt.axis("off")

        return fig
