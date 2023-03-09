import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from merlin.plots._base import AbstractPlot, PlotMetadata


class XDriftViolinPlot(AbstractPlot):
    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {"warp_task": "all"}

    def get_required_metadata(self):
        return [DriftCorrectionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        data = inputMetadata["driftplots/DriftCorrectionMetadata"].drifts
        fig = plt.figure()
        plt.axhline(y=0, linestyle=":", color="#bbbbbb")
        sns.violinplot(data=data[data["Round"] > 1], x="Round", y="x", scale="count")
        plt.ylabel("X drift")
        return fig


class YDriftViolinPlot(AbstractPlot):
    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {"warp_task": "all"}

    def get_required_metadata(self):
        return [DriftCorrectionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        data = inputMetadata["driftplots/DriftCorrectionMetadata"].drifts
        fig = plt.figure()
        plt.axhline(y=0, linestyle=":", color="#bbbbbb")
        sns.violinplot(data=data[data["Round"] > 1], x="Round", y="y", scale="count")
        plt.ylabel("Y drift")
        return fig


class DriftPathPlot(AbstractPlot):
    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {"warp_task": "all"}

    def get_required_metadata(self):
        return [DriftCorrectionMetadata]

    def _generate_plot(self, inputTasks, inputMetadata):
        data = inputMetadata["driftplots/DriftCorrectionMetadata"].drifts
        fig = plt.figure()
        path = data.groupby("Round").median()
        plt.plot(path["x"], path["y"], marker=None, linestyle=":", color="#777777")
        plt.scatter(path["x"], path["y"], c=range(len(path)), cmap="gist_rainbow", edgecolors="black", zorder=2)
        plt.xlabel("X drift")
        plt.ylabel("Y drift")
        return fig


class DriftCorrectionMetadata(PlotMetadata):
    def __init__(self, analysisTask, taskDict):
        super().__init__(analysisTask, taskDict)

        self.warp_task = self._taskDict["warp_task"]
        self.completed = {fov: False for fov in self.warp_task.fragment_list()}
        self.drifts = pd.DataFrame()

    def update(self) -> None:
        for fov in self.warp_task.fragment_list():
            if not self.completed[fov] and self.warp_task.is_complete(fov):
                drifts = pd.DataFrame(self.warp_task.get_transformation(fov), columns=["x", "y"])
                drifts["fov"] = fov
                drifts = drifts.reset_index().rename(columns={"index": "Round"})
                drifts["Round"] += 1
                self.drifts = pd.concat([self.drifts, drifts])
                self.completed[fov] = True

    def is_complete(self) -> bool:
        return all(self.completed.values())
