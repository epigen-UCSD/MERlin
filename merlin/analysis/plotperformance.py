import os
import time

from matplotlib import pyplot as plt

import merlin
from merlin import plots
from merlin.core import analysistask

plt.style.use(os.sep.join([os.path.dirname(merlin.__file__), "ext", "default.mplstyle"]))


class PlotPerformance(analysistask.AnalysisTask):
    """An analysis task that generates plots depicting metrics of the MERFISH decoding."""

    def setup(self) -> None:
        super().setup(parallel=False)

        self.set_default_parameters({
            "exclude_plots": []
        })

        self.taskTypes = [
            "warp_task",
            "decode_task",
            "filter_task",
            "optimize_task",
            "segment_task",
            "sum_task",
            "partition_task",
            "global_align_task",
            "output_task",
        ]

    def run_analysis(self):
        taskDict = {
            t: self.dataSet.load_analysis_task(self.parameters[t], "") for t in self.taskTypes if t in self.parameters
        }
        plotEngine = plots.PlotEngine(self, taskDict)
        while not plotEngine.take_step():
            time.sleep(2)
