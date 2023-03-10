import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


class DriftCorrectionMetadata(PlotMetadata):
    def __init__(self, plot_task, required_tasks):
        super().__init__(plot_task, required_tasks)

        self.warp_task = self.required_tasks["warp_task"]
        self.drifts = pd.DataFrame()
        self.register_updaters({"warp_task": self.process_update})

    def process_update(self, fragment):
        drifts = pd.DataFrame(self.warp_task.get_transformation(fragment), columns=["X drift", "Y drift"])
        drifts["fov"] = fragment
        drifts = drifts.reset_index().rename(columns={"index": "Round"})
        drifts["Round"] += 1
        self.drifts = pd.concat([self.drifts, drifts])
