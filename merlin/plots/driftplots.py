import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot, PlotMetadata


def make_drift_violin_plot(data, axis):
    fig = plt.figure()
    plt.axhline(y=0, linestyle=":", color="#bbbbbb")
    sns.violinplot(data=data[data["Round"] > 1], x="Round", y=axis, scale="count")
    return fig


class XDriftViolinPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"warp_task": "all"})
        self.set_required_metadata(DriftCorrectionMetadata)

    def create_plot(self, **kwargs):
        data = kwargs["metadata"]["driftplots/DriftCorrectionMetadata"].drifts
        return make_drift_violin_plot(data, "X drift")


class YDriftViolinPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"warp_task": "all"})
        self.set_required_metadata(DriftCorrectionMetadata)

    def create_plot(self, **kwargs):
        data = kwargs["metadata"]["driftplots/DriftCorrectionMetadata"].drifts
        return make_drift_violin_plot(data, "Y drift")


class ZDriftViolinPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"warp_task": "all"})
        self.set_required_metadata(DriftCorrectionMetadata)

    def create_plot(self, **kwargs):
        data = kwargs["metadata"]["driftplots/DriftCorrectionMetadata"].drifts
        if "Z drift" in data:
            return make_drift_violin_plot(data, "Z drift")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Z-drift was not adjusted", ha="center", va="center", transform=ax.transAxes)
        return fig


class DriftPathPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"warp_task": "all"})
        self.set_required_metadata(DriftCorrectionMetadata)

    def create_plot(self, **kwargs):
        data = kwargs["metadata"]["driftplots/DriftCorrectionMetadata"].drifts
        fig = plt.figure()
        path = data.groupby("Round")[["X drift", "Y drift"]].median()
        plt.plot(path["X drift"], path["Y drift"], marker=None, linestyle=":", color="#777777")
        plt.scatter(
            path["X drift"], path["Y drift"], c=range(len(path)), cmap="gist_rainbow", edgecolors="black", zorder=2
        )
        plt.xlabel("X drift")
        plt.ylabel("Y drift")
        return fig
    

class AlignedBitImagesPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"warp_task": "all"})
        self.formats = [".png"]

    def create_plot(self, **kwargs) -> plt.Figure:
        fragment = self.plot_task.dataSet.get_fovs()[0]
        align_task = self.plot_task.dataSet.load_analysis_task("FiducialAlign", fragment=fragment)
        imgs = align_task.get_aligned_image_set(fragment)
        colors = self.plot_task.dataSet.dataOrganization.get_data_colors()
        channels = self.plot_task.dataSet.dataOrganization.get_channels_for_color(colors[0])
        zpos = self.plot_task.dataSet.get_z_positions()
        zindex = self.plot_task.dataSet.position_to_z_index(zpos[len(zpos) // 2])

        nrows = int(np.ceil(len(channels) / 3))
        fig, ax = plt.subplots(nrows, 2, figsize=(14.5, 7*nrows), dpi=200)
        for i in range(nrows):
            ind = i*3
            inds = channels.index[ind:ind+3]

            img1 = self.plot_task.dataSet.get_raw_image(inds[0], fragment, zindex)
            if len(inds) > 1:
                img2 = self.plot_task.dataSet.get_raw_image(inds[1], fragment, zindex)
            else:
                img2 = np.zeros_like(img1)
            if len(inds) > 2:
                img3 = self.plot_task.dataSet.get_raw_image(inds[2], fragment, zindex)
            else:
                img3 = np.zeros_like(img1)

            color_img1 = np.moveaxis(np.array([
                img1 / np.percentile(img1, 99),
                img2 / np.percentile(img2, 99),
                img3 / np.percentile(img3, 99)
            ]), 0, -1)

            img1 = imgs[inds[0], zindex, :, :]
            if len(inds) > 1:
                img2 = imgs[inds[1], zindex, :, :]
            else:
                img2 = np.zeros_like(img1)
            if len(inds) > 2:
                img3 = imgs[inds[2], zindex, :, :]
            else:
                img3 = np.zeros_like(img1)

            color_img2 = np.moveaxis(np.array([
                img1 / np.percentile(img1, 99),
                img2 / np.percentile(img2, 99),
                img3 / np.percentile(img3, 99)
            ]), 0, -1)

            ax[i, 0].imshow(color_img1)
            ax[i, 0].axis("off")
            ax[i, 0].text(0, 1, channels[inds[0]], color="#ff0000", transform=ax[i,0].transAxes, ha="left", va="top")
            if len(inds) > 1:
                ax[i, 0].text(0, 0.97, channels[inds[1]], color="#00ff00", transform=ax[i, 0].transAxes, ha="left", va="top")
            if len(inds) > 2:
                ax[i, 0].text(0, 0.94, channels[inds[2]], color="#0000ff", transform=ax[i, 0].transAxes, ha="left", va="top")
            ax[i, 0].text(0.5, 1, "Raw images", color="#ffffff", transform=ax[i, 0].transAxes, ha="center", va="top")
            ax[i, 1].imshow(color_img2)
            ax[i, 1].text(0.5, 1, "Aligned images", color="#ffffff", transform=ax[i, 1].transAxes, ha="center", va="top")
            ax[i, 1].axis("off")
        fig.tight_layout()

        return fig


class DriftCorrectionMetadata(PlotMetadata):
    def __init__(self, plot_task, required_tasks):
        super().__init__(plot_task, required_tasks)

        self.warp_task = self.required_tasks["warp_task"]
        self.drifts = pd.DataFrame()
        self.register_updaters({"warp_task": self.process_update})

    def process_update(self, fragment):
        self.warp_task.fragment = fragment
        drifts = self.warp_task.get_transformation()
        columns = ["X drift", "Y drift"]
        if len(list(drifts.values())[0]) == 3:
            columns = ["Z drift"] + columns
        drifts = pd.DataFrame(drifts.values(), columns=columns)
        drifts["fov"] = fragment
        drifts = drifts.reset_index().rename(columns={"index": "Round"})
        drifts["Round"] += 1
        self.drifts = pd.concat([self.drifts, drifts])
