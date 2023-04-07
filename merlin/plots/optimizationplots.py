import seaborn as sns
from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot


class OptimizationScaleFactorsPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"optimize_task": "all"})

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(kwargs["tasks"]["optimize_task"].get_scale_factor_history())
        plt.xlabel("Bit index")
        plt.ylabel("Iteration number")
        plt.title("Scale factor optimization history")
        return fig


class ScaleFactorVsBitNumberPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"optimize_task": "all"})

    def create_plot(self, **kwargs):
        optimize_task = kwargs["tasks"]["optimize_task"]
        codebook = optimize_task.get_codebook()
        data_organization = optimize_task.dataSet.get_data_organization()
        colors = [
            data_organization.get_data_channel_color(data_organization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()
        ]

        scale_factors = optimize_task.load_result("scale_factors")
        scale_factors_by_color = {c: [] for c in set(colors)}
        for i, s in enumerate(scale_factors):
            scale_factors_by_color[colors[i]].append((i, s))

        fig = plt.figure(figsize=(5, 5))
        for d in scale_factors_by_color.values():
            plt.plot([x[0] for x in d], [x[1] for x in d], "o")

        plt.legend(scale_factors_by_color.keys())
        plt.ylim(bottom=0)
        plt.xlabel("Bit index")
        plt.ylabel("Scale factor magnitude")
        plt.title("Scale factor magnitude vs bit index")
        return fig


class OptimizationBarcodeCountsPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"optimize_task": "all"})

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(5, 5))
        sns.heatmap(kwargs["tasks"]["optimize_task"].get_barcode_count_history())
        plt.xlabel("Barcode index")
        plt.ylabel("Iteration number")
        plt.title("Barcode counts optimization history")
        return fig
