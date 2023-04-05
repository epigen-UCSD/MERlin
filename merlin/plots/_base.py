import contextlib
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from matplotlib import pyplot as plt

from merlin.core import analysistask


class AbstractPlot(ABC):
    """A base class for generating a plot of the analysis results.

    Each plot should inherit from this class.
    """

    def __init__(self, plot_task: analysistask.AnalysisTask) -> None:
        """Create a new AbstractPlot.

        Args:
            plot_task: the analysisTask where the plot should be saved.
        """
        self.plot_task = plot_task
        self.formats = [".png", ".pdf"]
        self.set_required_tasks({})
        self.set_required_metadata([])

    def figure_name(self) -> str:
        """Get the name for identifying this figure.

        Returns: the name of this figure
        """
        return type(self).__name__

    def set_required_tasks(self, required_tasks) -> None:
        """Set the tasks that are required to be complete prior to generating this plot.

        required_tasks: A dictionary of the types of tasks as keys and a tuple
            of the accepted classes as values. The keys can include
            decode_task, filter_task, optimize_task, segment_task,
            sum_task, partition_task, and/or global_align_task. If all classes
            of the specified type are allowed, the value should be 'all'. If
            no tasks are required then an empty dictionary should be returned.
        """
        self.required_tasks = required_tasks

    def set_required_metadata(self, *required_metadata) -> None:
        """Set the plot metadata that is required to generate this plot.

        required_metadata: A list of class references for the metadata
            objects that are required for this task.
        """
        self.required_metadata = required_metadata

    @abstractmethod
    def create_plot(self, **kwargs) -> plt.Figure:
        """Generate the plot.

        This function should be implemented in all subclasses and the generated
        figure handle should be returned.

        Keyword Args:
            tasks: A dictionary of the input tasks to required for the
                plot. Each analysis task is indexed by a string indicating
                the task type as in get_required_tasks.
            metadata: A dictionary of the input metadata for generating
                this plot. Each metadata object is indexed by the name of the
                metadata.
        Returns: the figure handle to the newly generated figure
        """

    def is_relevant(self, tasks: Dict[str, analysistask.AnalysisTask]) -> bool:
        """Determine if this plot is relevant given the analysis tasks provided.

        Args:
            tasks: A dictionary of the analysis tasks indexed with
                strings indicating the task type as in get_required_tasks
        Returns: True if this plot can be generated using the provided
            analysis tasks and false otherwise.
        """
        for req_task, req_types in self.required_tasks.items():
            if req_task not in tasks:
                return False
            if req_types != "all" and not isinstance(tasks[req_task], req_types):
                return False
        return True

    def is_ready(self, complete_tasks: List[str], complete_metadata: List[str]) -> bool:
        """Determine if all requirements for generating this plot are satisfied.

        Args:
            complete_tasks: A list of the types of tasks that are complete.
                The list can contain the same strings as in get_required_tasks
            complete_metadata: A list of the metadata that has been generated.
        Returns: True if all required tasks and all required metadata
            is complete
        """
        return all([t in complete_tasks for t in self.required_tasks]) and all(
            [m.metadata_name() in complete_metadata for m in self.required_metadata if m]
        )

    def is_complete(self) -> bool:
        """Determine if this plot has been generated.

        Returns: True if this plot has been generated and otherwise false.
        """
        return self.plot_task.dataSet.figure_exists(
            self.plot_task, self.figure_name(), type(self).__module__.split(".")[-1]
        )

    def plot(self, tasks: Dict[str, analysistask.AnalysisTask], metadata: Dict[str, "PlotMetadata"]) -> None:
        """Generate this plot and save it within the analysis task.

        If the plot is not relevant for the types of analysis tasks passed,
        then the function will return without generating any plot.

        Args:
            tasks: A dictionary of the input tasks to use to generate the
                plot. Each analysis task is indexed by a string indicating
                the task type as in get_required_tasks.
            metadata: A dictionary of the input metadata for generating
                this plot. Each metadata object is indexed by the name of the
                metadata.
        """
        if not self.is_relevant(tasks):
            return
        f = self.create_plot(tasks=tasks, metadata=metadata)
        if f:
            f.tight_layout(pad=1)
            self.plot_task.dataSet.save_figure(
                self.plot_task, f, self.figure_name(), type(self).__module__.split(".")[-1], formats=self.formats
            )
            plt.close(f)


class PlotMetadata:
    """A class for collecting metadata needed for plots from tasks.

    If the metadata needed for a plot is computationally intensive, this class can be used to
    process the results from each parallel instance of a task as they are completed so that
    the computation can be spread out while other tasks are running.

    Once a class inherits this class, the __init__ function of the new class should call
    `self.register_updaters` to designate callback functions that will be used when a task
    is completed. For example, if a subclass calls `self.register_updaters({"decode_task": self.process_barcodes})`
    then whenever the decoding is done on a FOV, the `process_barcodes` method of the subclass
    will be called and passed the FOV name that completed.

    If a subclass wishes to incrementally save metadata as they are processed, the
    `self.register_datasets` function can be used. As an example, if a subclass was
    using `self.computed_results` to store some metadata, by calling
    `self.register_updaters(["computed_results"])` in __init__, this metadata would
    then be saved automatically on each update, and loaded if MERlin is restarted.
    """

    def __init__(self, plot_task: analysistask.AnalysisTask, required_tasks: Dict[str, analysistask.AnalysisTask]):
        """Create a new metadata object.

        Args:
            plot_task: the analysisTask where the metadata should be saved.
            required_tasks: a dictionary containing the analysis tasks to use
                to generate the metadata indexed by the type of task as a
                string as in get_required_tasks
        """
        self.plot_task = plot_task
        self.required_tasks = required_tasks
        self.datasets = []
        self.completed = {}

    @classmethod
    def metadata_name(cls) -> str:
        return cls.__module__.split(".")[-1] + "/" + cls.__name__

    def save_metadata(self, dataset: str) -> None:
        metadata = getattr(self, dataset)
        if isinstance(metadata, np.ndarray):
            self.plot_task.dataSet.save_numpy_analysis_result(
                metadata, dataset, self.plot_task, subdirectory=self.metadata_name()
            )
        else:
            self.plot_task.dataSet.save_pickle_analysis_result(
                metadata, dataset, self.plot_task, subdirectory=self.metadata_name()
            )

    def load_metadata(self, dataset: str) -> Any:
        path = self.plot_task.dataSet._analysis_result_save_path("", self.plot_task, subdirectory=self.metadata_name())

        files = list(pathlib.Path(path).glob(f"{dataset}.*"))
        if not files:
            raise FileNotFoundError
        data_format = files[0].suffix
        if data_format == ".npy":
            return self.plot_task.dataSet.load_numpy_analysis_result(
                dataset, self.plot_task, subdirectory=self.metadata_name()
            )
        if data_format == ".pkl":
            return self.plot_task.dataSet.load_pickle_analysis_result(
                dataset, self.plot_task, subdirectory=self.metadata_name()
            )
        raise FileNotFoundError

    def save_state(self) -> None:
        if not self.datasets:
            return
        try:
            for dataset in self.datasets:
                self.save_metadata(dataset)
        except AttributeError:
            pass
        else:  # Only save progress if all datasets were saved
            self.plot_task.dataSet.save_pickle_analysis_result(
                self.completed, "completed", self.plot_task, subdirectory=self.metadata_name()
            )

    def load_state(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.completed = self.plot_task.dataSet.load_pickle_analysis_result(
                "completed", self.plot_task, subdirectory=self.metadata_name()
            )
            for dataset in self.datasets:
                setattr(self, dataset, self.load_metadata(dataset))

    def register_updaters(self, updaters) -> None:
        self.updaters = updaters
        for task in updaters:
            self.completed[task] = {fragment: False for fragment in self.required_tasks[task].fragment_list}

    def register_datasets(self, *datasets) -> None:
        self.datasets = datasets

    def update(self) -> None:
        """Update this metadata with the latest analysis results."""
        updated = False
        for task, completed in self.completed.items():
            for fragment, done in completed.items():
                if not done and self.required_tasks[task].is_complete():
                    self.updaters[task](fragment)
                    self.completed[task][fragment] = True
                    updated = True
        if updated:
            self.save_state()

    def is_complete(self) -> bool:
        """Determine if this metadata is complete.

        Returns: True if the metadata is complete or False if additional
            computation is necessary
        """
        return self.completed and all([all(completed.values()) for completed in self.completed.values()])

    def num_completed(self) -> int:
        return sum([sum(completed.values()) for completed in self.completed.values()])
