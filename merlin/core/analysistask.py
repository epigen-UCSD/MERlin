import copy
import cProfile
import io
import pstats
import threading
from typing import Any

import merlin


class AnalysisAlreadyStartedError(Exception):
    """Analysis has already started."""


class AnalysisAlreadyExistsError(Exception):
    """Analysis already exists."""


class InvalidParameterError(Exception):
    """Analysis parameters are invalid."""


class AnalysisTask:
    """An abstract class for performing analysis on a DataSet.

    Subclasses should implement the analysis to perform in the run_analysis() function.
    """

    def __init__(self, dataSet, parameters: dict[str, Any], analysis_name: str, *, parallel: bool = False) -> None:
        """Create an AnalysisTask object that performs analysis on the specified DataSet.

        Args:
            dataSet: the DataSet to run analysis on.
            parameters: a dictionary containing parameters used to run the
                analysis.
            analysisName: specifies a unique identifier for this
                AnalysisTask. If analysisName is not set, the analysis name
                will default to the name of the class.
        """
        self.dataSet = dataSet
        self.parameters = {} if parameters is None else copy.deepcopy(parameters)
        if analysis_name is None:
            self.analysis_name = type(self).__name__
        else:
            self.analysis_name = analysis_name

        if "merlin_version" not in self.parameters:
            self.parameters["merlin_version"] = merlin.version()
        elif not merlin.is_compatible(self.parameters["merlin_version"]):
            raise merlin.IncompatibleVersionError(
                (
                    "Analysis task %s has already been created by MERlin "
                    "version %s, which is incompatible with the current "
                    "MERlin version, %s"
                )
                % (self.analysis_name, self.parameters["merlin_version"], merlin.version())
            )

        self.parameters["module"] = type(self).__module__
        self.parameters["class"] = type(self).__name__

        if "codebookNum" in self.parameters:
            self.codebookNum = self.parameters["codebookNum"]

        self._fragment_list = self.dataSet.get_fovs() if parallel else []
        self.dependencies = set()

    def __getattr__(self, attr):
        """Check if an unloaded dependency is being accessed and load it."""
        if attr in self.dependencies:
            task = self.dataSet.load_analysis_task(self.parameters[attr])
            setattr(self, attr, task)
            return task
        raise AttributeError(f"{attr} not an attribute of {self}")

    def add_dependencies(self, *args, optional: bool = False) -> None:
        """Add the given dependencies.

        If optional is True, only dependencies specified in self.parameters are
        added.
        """
        if optional:
            self.dependencies.update(task for task in args if task in self.parameters)
        else:
            self.dependencies.update(args)

    @property
    def fragment_list(self) -> list[str]:
        """Get the list of fragments (FOVs) this task is run on.

        This should be set by specific analysis tasks that subclass this class. By default,
        the fragment list is empty indicating that the task is not run in parallel on each
        FOV.
        """
        return self._fragment_list

    @fragment_list.setter
    def fragment_list(self, value: list[str]) -> None:
        self._fragment_list = value

    def save(self, *, overwrite: bool = False) -> None:
        """Save a copy of this AnalysisTask into the data set.

        Args:
            overwrite: flag indicating if an existing analysis task with the
                same name as this analysis task should be overwritten even
                if the specified parameters are different.

        Raises:
            AnalysisAlreadyExistsException: if an analysis task with the
                same name as this analysis task already exists in the
                data set with different parameters.
        """
        self.dataSet.save_analysis_task(self, overwrite)

    def run(self, fragment: str = "", *, overwrite: bool = True) -> None:
        """Run this AnalysisTask.

        Upon completion of the analysis, this function informs the DataSet
        that analysis is complete.

        Args:
            overwrite: flag indicating if previous analysis from this
                analysis task should be overwritten.

        Raises:
            AnalysisAlreadyStartedError: if this analysis task is currently
                already running or if overwrite is not True and this analysis
                task has already completed or exited with an error.
        """
        if not fragment and self.is_parallel():
            for i in self.fragment_list:
                self.run(i, overwrite=overwrite)
        else:
            logger = self.dataSet.get_logger(self, fragment)
            logger.info(f"Beginning {self.analysis_name} {fragment}")
            try:
                if self.is_running(fragment):
                    raise AnalysisAlreadyStartedError(
                       f"Unable to run {self.analysis_name} fragment {fragment} since it is already running"
                    )

                if overwrite:
                    self.reset_analysis(fragment)

                if self.is_complete(fragment) or self.is_error(fragment):
                    raise AnalysisAlreadyStartedError(
                       f"Unable to run {self.analysis_name} fragment {fragment} since it has already run"
                    )

                self.dataSet.record_analysis_started(self, fragment)
                self.indicate_running(fragment)
                if self.dataSet.profile:
                    profiler = cProfile.Profile()
                    profiler.enable()
                if fragment:
                    self.run_analysis(fragment=fragment)
                else:
                    self.run_analysis()
                if self.dataSet.profile:
                    profiler.disable()
                    stat_string = io.StringIO()
                    stats = pstats.Stats(profiler, stream=stat_string)
                    stats.sort_stats("time")
                    stats.print_stats()
                    logger.info(stat_string.getvalue())
                self.dataSet.record_analysis_complete(self, fragment)
                logger.info(f"Completed {self.analysis_name} {fragment}")
                self.dataSet.close_logger(self, fragment)
            except Exception:
                logger.exception("")
                self.dataSet.record_analysis_error(self, fragment)
                self.dataSet.close_logger(self, fragment)

    def reset_analysis(self, fragment: str = "") -> None:
        """Remove files created by this analysis task and remove markers
        indicating that this analysis has been started, or has completed.

        This function should be overridden by subclasses so that they
        can delete the analysis files.
        """
        if not fragment and self.is_parallel():
            for i in self.fragment_list:
                self.reset_analysis(i)
        self.dataSet.reset_analysis_status(self)

    def indicate_running(self, fragment: str = "") -> None:
        """A loop that regularly signals to the dataset that this analysis
        task is still running successfully.

        Once this function is called, the dataset will be notified every
        minute that this analysis is still running until the analysis
        completes.
        """
        if self.is_complete(fragment) or self.is_error(fragment):
            return

        self.dataSet.record_analysis_running(self, fragment)
        self.runTimer = threading.Timer(30, self.indicate_running, [fragment])
        self.runTimer.daemon = True
        self.runTimer.start()

    def is_error(self, fragment: str = "") -> bool:
        """Determine if an error has occurred while running this analysis."""
        if not fragment and self.is_parallel():
            return any(self.is_error(i) for i in self.fragment_list)
        return self.dataSet.check_analysis_error(self, fragment)

    def is_complete(self, fragment: str = "") -> bool:
        """Determine if this analysis has completed successfully."""
        if self.is_parallel() and not fragment:
            if self.dataSet.check_analysis_done(self):
                return True
            all_complete = all(self.is_complete(i) for i in self.fragment_list)
            if all_complete:
                self.dataSet.record_analysis_complete(self)
                return True
            return False
        return self.dataSet.check_analysis_done(self, fragment)

    def is_started(self, fragment: str = "") -> bool:
        """Determine if this analysis has started."""
        if self.is_parallel() and not fragment:
            return any(self.is_started(i) for i in self.fragment_list)
        return self.dataSet.check_analysis_started(self, fragment)

    def is_running(self, fragment: str = "") -> bool:
        """Determines if this analysis task is expected to be running,
        but has unexpectedly stopped for more than two minutes.
        """
        if not self.is_started(fragment):
            return False
        if self.is_complete(fragment):
            return False

        return not self.dataSet.is_analysis_idle(self, fragment)

    def is_parallel(self) -> bool:
        """Determine if this analysis task uses multiple cores."""
        return len(self.fragment_list) > 0
