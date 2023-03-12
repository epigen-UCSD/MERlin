import matplotlib.pyplot as plt

from merlin.plots import tools
from merlin.plots._base import AbstractPlot


class MedianTranscriptsPerCellPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"output_task": "all"})

    def create_plot(self, **kwargs):
        adata = kwargs["tasks"]["output_task"].get_scanpy_object()
        fig = tools.plot_histogram(adata.obs, "total_counts")
        plt.xlabel("Transcripts per cell")
        return fig


class MedianGenesPerCellPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"output_task": "all"})

    def create_plot(self, **kwargs):
        adata = kwargs["tasks"]["output_task"].get_scanpy_object()
        fig = tools.plot_histogram(adata.obs, "n_genes_by_counts")
        plt.xlabel("Transcripts per cell")
        return fig
