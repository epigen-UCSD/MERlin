import numpy as np
from matplotlib import pyplot as plt

from merlin.analysis.output import FinalOutput
from merlin.analysis.segment import CellposeSegment, FeatureSavingAnalysisTask
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
        self.set_required_tasks({"segment_task": CellposeSegment, "output_task": FinalOutput})
        self.formats = [".png"]

    def plot_mask(self, fov, ax) -> None:
        mask = self.segment_task.load_mask(fov)
        channel = self.dataset.get_data_organization().get_data_channel_index(self.segment_task.parameters["channel"])
        if self.segment_task.parameters["z_pos"] is not None:
            z_index = self.dataset.position_to_z_index(self.segment_task.parameters["z_pos"])
            image = self.dataset.get_raw_image(channel, fov, z_index)
        else:
            z_positions = self.dataset.get_z_positions()
            z_index = z_positions[len(z_positions) // 2]
            image = self.dataset.get_raw_image(channel, fov, z_index)

        ax.imshow(image, cmap="gray")
        ax.contour(
            mask[int(z_index)],
            [x + 0.5 for x in np.unique(mask[int(z_index)])],
            colors="tab:blue",
            linewidths=1,
            zorder=2,
        )
        ax.contourf(mask[int(z_index)], [x + 0.5 for x in np.unique(mask[int(z_index)])], colors="tab:blue", alpha=0.2)
        ax.axis("off")

    def create_plot(self, **kwargs) -> plt.Figure:
        self.segment_task = kwargs["tasks"]["segment_task"]
        self.dataset = self.segment_task.dataSet
        output_task = kwargs["tasks"]["output_task"]
        metadata = output_task.get_cell_metadata_table()
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
        self.set_required_tasks({"output_task": FinalOutput})

    def create_plot(self, **kwargs) -> plt.Figure:
        output_task = kwargs["tasks"]["output_task"]
        metadata = output_task.get_cell_metadata_table()
        fig = tools.plot_histogram(metadata, "volume")
        plt.xlabel("Cell volume (pixels)")
        return fig
