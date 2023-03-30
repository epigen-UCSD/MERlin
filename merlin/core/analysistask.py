import copy
import cProfile
import io
import json
import os
import pickle
import pstats
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc

import merlin


class AnalysisAlreadyStartedError(Exception):
    """Analysis has already started."""


class AnalysisAlreadyExistsError(Exception):
    """Analysis already exists."""


class InvalidParameterError(Exception):
    """Analysis parameters are invalid."""


class ResultNotFoundError(Exception):
    """The analysis result is not found."""


class AnalysisTask:
    """An abstract class for performing analysis on a DataSet.

    Subclasses should implement the analysis to perform in the run_analysis() function.
    """

    def __init__(self, dataSet, path: Path, parameters: dict[str, Any], analysis_name: str) -> None:
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

        self.path = path / self.analysis_name

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

        self.setup()  # Will give an error if setup() isn't implemented in a subclass, this is intentional.

    def setup(self, *, parallel: bool) -> None:
        self._fragment_list = self.dataSet.get_fovs() if parallel else []
        self.dependencies = set()
        self.results = {}

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

    def define_results(self, *metadata) -> None:
        for result in metadata:
            if isinstance(result, str):
                result_name = result
                kwargs = {}
            elif isinstance(result, tuple):
                result_name, kwargs = result
            self.results[result_name] = kwargs
            if self.is_parallel():
                Path(self.path, result_name).mkdir(parents=True, exist_ok=True)

    def set_default_parameters(self, defaults: dict[str, Any]) -> None:
        for key, value in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

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

                if self.is_complete(fragment):
                    raise AnalysisAlreadyStartedError(
                        f"Unable to run {self.analysis_name} fragment {fragment} since it has already run"
                    )

                self.record_status("start", fragment)
                self.record_environment(fragment)
                self.indicate_running(fragment)
                if self.dataSet.profile:
                    profiler = cProfile.Profile()
                    profiler.enable()
                if fragment:
                    self.run_analysis(fragment=fragment)
                    self.save_results(fragment)
                else:
                    self.run_analysis()
                    self.save_results()
                if self.dataSet.profile:
                    profiler.disable()
                    stat_string = io.StringIO()
                    stats = pstats.Stats(profiler, stream=stat_string)
                    stats.sort_stats("time")
                    stats.print_stats()
                    logger.info(stat_string.getvalue())
                self.record_status("done", fragment)
                logger.info(f"Completed {self.analysis_name} {fragment}")
                self.dataSet.close_logger(self, fragment)
            except Exception:
                logger.exception("")
                self.record_status("error", fragment)
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
        self.reset()

    def indicate_running(self, fragment: str = "") -> None:
        """A loop that regularly signals to the dataset that this analysis
        task is still running successfully.

        Once this function is called, the dataset will be notified every
        minute that this analysis is still running until the analysis
        completes.
        """
        if self.is_complete(fragment) or self.is_error(fragment):
            return

        self.record_status("run", fragment)
        self.runTimer = threading.Timer(30, self.indicate_running, [fragment])
        self.runTimer.daemon = True
        self.runTimer.start()

    def status_file(self, status: str, fragment: str = "") -> Path:
        filename = f"{self.analysis_name}_{fragment}.{status}" if fragment else f"{self.analysis_name}.{status}"
        return Path(self.path, "tasks", filename)

    def status(self, status: str, fragment: str = "") -> bool:
        return self.status_file(status, fragment).exists()

    def record_status(self, status: str, fragment: str = "") -> None:
        filename = self.status_file(status, fragment)
        with filename.open("w") as f:
            f.write(f"{time.time()}")

    def reset_status(self, status: str, fragment: str = "") -> None:
        filename = self.status_file(status, fragment)
        filename.unlink(missing_ok=True)

    def reset(self, fragment: str = "") -> None:
        if self.is_running():
            raise AnalysisAlreadyStartedError()

        self.reset_status("start", fragment)
        self.reset_status("run", fragment)
        self.reset_status("done", fragment)
        self.reset_status("error", fragment)
        self.reset_status("done")

    def record_environment(self, fragment: str = "") -> None:
        filename = self.status_file("environment", fragment)
        with filename.open("w") as outfile:
            json.dump(dict(os.environ), outfile, indent=4)

    def get_environment(self, fragment: str = "") -> None:
        """Get the environment variables for the system used to run the
        specified analysis task.

        Args:
            analysisTask: The completed analysis task to get the environment
                variables for.
            fragmentIndex: The fragment index of the analysis task to
                get the environment variables for.

        Returns: A dictionary of the environment variables. If the job has not
            yet run, then None is returned.
        """
        if not self.status("done", fragment):
            return None

        filename = self.status_file("environment", fragment)
        with filename.open() as infile:
            return json.load(infile)

    def is_error(self, fragment: str = "") -> bool:
        """Determine if an error has occurred while running this analysis."""
        if not fragment and self.is_parallel():
            return any(self.is_error(i) for i in self.fragment_list)
        return self.status("error", fragment)

    def is_complete(self, fragment: str = "") -> bool:
        """Determine if this analysis has completed successfully."""
        if self.is_parallel() and not fragment:
            if self.status("done"):
                return True
            all_complete = all(self.is_complete(i) for i in self.fragment_list)
            if all_complete:
                self.record_status("done")
                return True
            return False
        return self.status("done", fragment)

    def is_started(self, fragment: str = "") -> bool:
        """Determine if this analysis has started."""
        if self.is_parallel() and not fragment:
            return any(self.is_started(i) for i in self.fragment_list)
        return self.status("start", fragment)

    def is_running(self, fragment: str = "") -> bool:
        """Determines if this analysis task is expected to be running,
        but has unexpectedly stopped for more than two minutes.
        """
        if not self.is_started(fragment):
            return False
        if self.is_complete(fragment):
            return False

        return not self.is_idle(fragment)

    def is_idle(self, fragment: str = "") -> bool:
        filename = self.status_file("run", fragment)
        try:
            return time.time() - os.path.getmtime(filename) > 1
        except FileNotFoundError:
            return True

    def is_parallel(self) -> bool:
        """Determine if this analysis task uses multiple cores."""
        return len(self.fragment_list) > 0

    def result_path(self, result_name: str, extension: str, fragment: str = "") -> Path:
        if fragment:
            return self.path / result_name / f"{result_name}_{fragment}{extension}"
        return self.path / f"{result_name}{extension}"

    def save_results(self, fragment: str = "") -> None:
        for result_name in self.results:
            self.save_result(result_name, fragment)

    def save_result(self, result_name: str, fragment: str = "") -> None:
        if not hasattr(self, result_name):
            raise ResultNotFoundError(result_name)
        result = getattr(self, result_name)
        if isinstance(result, np.ndarray):
            np.save(self.result_path(result_name, ".npy", fragment), result)
        elif isinstance(result, pd.DataFrame):
            result.to_csv(self.result_path(result_name, ".csv", fragment), **self.results[result_name])
        elif isinstance(result, sc.AnnData):
            result.write(self.result_path(result_name, ".h5ad", fragment))
        else:
            with self.result_path(result_name, ".pkl", fragment).open("wb") as f:
                pickle.dump(result, f)

    def load_result(self, result_name: str, fragment: str = "") -> Any:
        if fragment:
            files = list(Path(self.path, result_name).glob(f"{result_name}_{fragment}.*"))
        else:
            files = list(self.path.glob(f"{result_name}.*"))
        filename = files[0]
        if filename.suffix == ".npy":
            return np.load(filename, allow_pickle=True)
        if filename.suffix == ".csv":
            if "index" not in self.results[result_name] or not self.results[result_name]["index"]:
                index_col = None
            else:
                index_col = 0
            return pd.read_csv(filename, index_col=index_col)
        if filename.suffix == ".h5ad":
            return sc.read(filename)
        if filename.suffix == ".pkl":
            with filename.open("rb") as f:
                return pickle.load(f)
        raise ResultNotFoundError(f"Unsupported format of {filename}")
