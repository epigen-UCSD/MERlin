import copy
import cProfile
import io
import json
import os
import pickle
import pstats
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc

import merlin


class ResultNotFoundError(Exception):
    """The analysis result is not found."""


class AnalysisTask:
    """An abstract class for performing analysis on a DataSet.

    Subclasses should implement the analysis to perform in the run_analysis() function.
    """

    def __init__(self, dataSet, path: Path, parameters: dict[str, Any], analysis_name: str, fragment: str) -> None:
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
        self.fragment = fragment

    def setup(self, *, parallel: bool) -> None:
        self._fragment_list = list(self.dataSet.get_fovs()) if parallel else []
        self.dependencies = {}
        self.results = {}
        self.final_results = {}

    def __getattr__(self, attr):
        """Check if an unloaded dependency is being accessed and load it."""
        if attr in self.dependencies:
            if isinstance(self.parameters[attr], str):
                task = self.dataSet.load_analysis_task(self.parameters[attr], "")
                if task.is_parallel() and self.fragment in task.fragment_list:
                    task.fragment = self.fragment
            else:
                task = []
                for t in self.parameters[attr]:
                    task.append(self.dataSet.load_analysis_task(t, ""))
                    if task[-1].is_parallel() and self.fragment in task[-1].fragment_list:
                        task[-1].fragment = self.fragment
            setattr(self, attr, task)
            return task
        raise AttributeError(f"{attr} not an attribute of {self}")

    def add_dependencies(self, dependencies: dict[str, list[str]], *, optional: bool = False) -> None:
        """Add the given dependencies.

        If optional is True, only dependencies specified in self.parameters are
        added.
        """
        if optional:
            self.dependencies.update({task: files for task, files in dependencies.items() if task in self.parameters})
        else:
            self.dependencies.update(dependencies)

    def define_results(self, *metadata, final: bool = False) -> None:
        for result in metadata:
            if isinstance(result, str):
                result_name = result
                kwargs = {}
            elif isinstance(result, tuple):
                result_name, kwargs = result
            else:
                continue
            if final:
                self.final_results[result_name] = kwargs
            else:
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

    @property
    def fragment(self) -> str:
        return self._fragment

    @fragment.setter
    def fragment(self, value: str) -> None:
        assert not value or (self.is_parallel() and value in self.fragment_list)
        self._fragment = value

    def has_finalize_step(self) -> bool:
        return hasattr(self, "finalize_analysis") or (self.is_parallel() and hasattr(self, "metadata"))

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

    def run(self, *, overwrite: bool = True) -> None:
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
        logger = self.dataSet.get_logger(self, self.fragment)
        logger.info(f"Beginning {self.analysis_name} {self.fragment}")
        try:
            if overwrite:
                self.reset_analysis()
            self.record_environment()
            if self.dataSet.profile:
                profiler = cProfile.Profile()
                profiler.enable()
            self.execute_task()
            if self.dataSet.profile:
                profiler.disable()
                stat_string = io.StringIO()
                stats = pstats.Stats(profiler, stream=stat_string)
                stats.sort_stats("time")
                stats.print_stats()
                logger.info(stat_string.getvalue())
            self.record_status("done")
            logger.info(f"Completed {self.analysis_name} {self.fragment}")
            self.dataSet.close_logger(self, self.fragment)
        except Exception:
            logger.exception("")
            self.dataSet.close_logger(self, self.fragment)

    def execute_task(self):
        if self.is_parallel() and not self.fragment:
            if hasattr(self, "finalize_analysis"):
                self.finalize_analysis()
                self.save_result()
            if hasattr(self, "metadata"):
                self.aggregate_metadata()
        else:
            self.run_analysis()
            if hasattr(self, "metadata"):
                self.save_metadata()
            self.save_result()

    def reset_analysis(self) -> None:
        """Remove files created by this analysis task and remove markers
        indicating that this analysis has been started, or has completed.

        This function should be overridden by subclasses so that they
        can delete the analysis files.
        """
        # if not self.fragment and self.is_parallel():
        #    for i in self.fragment_list:
        #        self.reset_analysis(i)
        # self.reset()
        ...

    def status_file(self, status: str, fragment: str = "") -> Path:
        if not fragment:
            fragment = self.fragment
        filename = f"{self.analysis_name}_{fragment}.{status}" if fragment else f"{self.analysis_name}.{status}"
        return Path(self.path, "tasks", filename)

    def status(self, status: str) -> bool:
        return self.status_file(status).exists()

    def record_status(self, status: str) -> None:
        filename = self.status_file(status)
        with filename.open("w") as f:
            f.write(f"{time.time()}")

    def record_environment(self) -> None:
        filename = self.status_file("environment")
        with filename.open("w") as outfile:
            json.dump(dict(os.environ), outfile, indent=4)

    def get_environment(self) -> None:
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
        if not self.status("done"):
            return None

        filename = self.status_file("environment")
        with filename.open() as infile:
            return json.load(infile)

    def is_complete(self) -> bool:
        """Determine if this analysis has completed successfully."""
        if self.is_parallel() and not self.fragment:
            if self.status("done"):
                return True
            if self.has_finalize_step():
                return False
            all_complete = all(
                self.dataSet.load_analysis_task(self.analysis_name, i).is_complete() for i in self.fragment_list
            )
            if all_complete:
                self.record_status("done")
                return True
            return False
        return self.status("done")

    def is_parallel(self) -> bool:
        """Determine if this analysis task uses multiple cores."""
        return len(self.fragment_list) > 0

    def result_path(self, result_name: str, extension: str) -> Path:
        if not extension.startswith("."):
            extension = f".{extension}"
        if self.fragment:
            return self.path / result_name / f"{result_name}_{self.fragment}{extension}"
        return self.path / f"{result_name}{extension}"

    def save_result(self, result_name: str = "") -> None:
        results = self.final_results if self.has_finalize_step() and not self.fragment else self.results
        if not result_name:
            for result_name in results:
                self.save_result(result_name)
        else:
            if not hasattr(self, result_name):
                raise ResultNotFoundError(result_name)
            result = getattr(self, result_name)
            if isinstance(result, np.ndarray):
                np.save(self.result_path(result_name, ".npy"), result)
            elif isinstance(result, pd.DataFrame):
                result.to_csv(self.result_path(result_name, ".csv"), **results[result_name])
            elif isinstance(result, sc.AnnData):
                result.write(self.result_path(result_name, ".h5ad"))
            else:
                with self.result_path(result_name, ".pkl").open("wb") as f:
                    pickle.dump(result, f)

    def load_result(self, result_name: str, fragment: str = "") -> Any:
        if not fragment:
            fragment = self.fragment
        results = self.final_results if self.has_finalize_step() and not fragment else self.results
        if fragment:
            files = list(Path(self.path, result_name).glob(f"{result_name}_{fragment}.*"))
        else:
            files = list(self.path.glob(f"{result_name}.*"))
        filename = files[0]
        if filename.suffix == ".npy":
            return np.load(filename, allow_pickle=True)
        if filename.suffix == ".csv":
            if "index" not in results[result_name] or not results[result_name]["index"]:
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

    def aggregate_result(self, result_name: str) -> list[Any]:
        return [self.load_result(result_name, fragment) for fragment in self.fragment_list]

    def save_metadata(self) -> None:
        path = self.result_path("metadata", ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.metadata(), f, indent=4, default=float)

    def aggregate_metadata(self) -> None:
        def combine_metadata(metadata, aggdata, prefix=""):
            for k, v in metadata.items():
                if isinstance(v, dict):
                    combine_metadata(v, aggdata, prefix=f"{k}/")
                else:
                    aggdata[prefix + k].append(v)

        metadata = defaultdict(list)
        for fragment in self.fragment_list:
            self.fragment = fragment
            with self.result_path("metadata", ".json").open() as f:
                combine_metadata(json.load(f), metadata)

        aggdata = {}
        for k, v in metadata.items():
            data = aggdata
            for token in k.split("/"):
                if token not in data:
                    data[token] = {}
                data = data[token]
            data["min"] = np.min(v)
            data["max"] = np.max(v)
            data["mean"] = np.mean(v)
            data["median"] = np.median(v)
            data["std"] = np.std(v)

        self.fragment = ""
        with self.result_path("metadata", ".json").open("w") as f:
            json.dump(aggdata, f, indent=4, default=float)
