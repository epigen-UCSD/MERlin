import numpy as np
from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot, PlotMetadata


class TestPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"test_task": "all"})
        self.set_required_metadata(TestPlotMetadata)

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(10, 10))
        plt.plot(kwargs["metadata"]["testplots/TestPlotMetadata"].get_mean_values(), "x")
        return fig


class TestPlotMetadata(PlotMetadata):
    def __init__(self, plot_task, required_tasks):
        super().__init__(plot_task, required_tasks)
        self.test_task = self.required_tasks["test_task"]
        self.meanValues = np.zeros(len(self.test_task.fragment_list()))
        self.register_updaters({"test_task": self.process})

    def get_mean_values(self) -> np.ndarray:
        return self.meanValues

    def process(self, fragment) -> None:
        self.meanValues[fragment] = np.mean(self.test_task.get_random_result(fragment))
