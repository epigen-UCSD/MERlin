import datetime
import functools
import importlib
import json
import logging
import math
import os
import pickle
import shutil
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import networkx as nx
import numpy as np
import pandas
import scanpy as sc
import tables
import tifffile
import xmltodict
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

import merlin
from merlin.core import analysistask
from merlin.data import codebook, dataorganization
from merlin.util import dataportal, imagereader

TaskOrName = Union[analysistask.AnalysisTask, str]


class DataFormatException(Exception):
    pass


class DataSet:
    def __init__(
        self,
        experiment_name: Path,
        data_root: Path | None = None,
        analysis_root: Path | None = None,
        analysis_suffix: str | None = None,
    ) -> None:
        """Create a dataset for the specified raw data.

        Args:
            experiment_name: the relative directory to the raw data
            data_root: the base path to the data. The data is expected
                to be in data_root/experiment_name. If data_root
                is not specified, DATA_HOME is read from the
                .env file.
            analysis_root: the base path for storing analysis results. Analysis
                results for this DataSet will be stored in
                analysis_root/experiment_name. If analysis_root is not
                specified, ANALYSIS_HOME is read from the .env file.
            analysis_suffix: A suffix to append to experiment_name when creating
                the analysis output folder. Enables different analyses on the
                same data.
        """
        if data_root is None:
            data_root = merlin.DATA_HOME
        if analysis_root is None:
            analysis_root = merlin.ANALYSIS_HOME

        self.experiment_name = experiment_name
        self.data_root = data_root
        self.analysis_root = analysis_root
        self.analysis_suffix = analysis_suffix

        self.raw_data_path = data_root / experiment_name
        self.raw_data_portal = dataportal.DataPortal.create_portal(self.raw_data_path)
        if not self.raw_data_portal.is_available():
            print(f"The raw data is not available at {self.raw_data_path}")

        analysis_name = Path(f"{str(experiment_name)}_{analysis_suffix}") if analysis_suffix else experiment_name
        self.analysis_path = analysis_root / analysis_name

        self._store_dataset_metadata()

        self.analysis_path.mkdir(parents=True, exist_ok=True)

    def _store_dataset_metadata(self) -> None:
        try:
            old_metadata = self.load_json_analysis_result("dataset", None)
            if not merlin.is_compatible(old_metadata["merlin_version"]):
                raise merlin.IncompatibleVersionError(
                    (
                        "Analysis was performed on dataset %s with MERlin "
                        "version %s, which is not compatible with the current "
                        "MERlin version %s"
                    )
                    % (self.experiment_name, old_metadata["version"], merlin.version())
                )
        except FileNotFoundError:
            new_metadata = {
                "merlin_version": merlin.version(),
                "module": type(self).__module__,
                "class": type(self).__name__,
                "dataset_name": self.experiment_name,
                "creation_date": str(datetime.datetime.now()),
                "analysis_path": str(self.analysis_path),
            }
            self.analysis_path.mkdir(parents=True, exist_ok=True)
            self.save_json_analysis_result(new_metadata, "dataset", None)

    def save_workflow(self, workflow_string: str) -> Path:
        """Save a snakemake workflow for analysis of this dataset.

        Args:
            workflow_string: a string containing the snakemake workflow
                to save

        Returns: the path to the saved workflow
        """
        snakemake_path = self.get_snakemake_path()
        snakemake_path.mkdir(parents=True, exist_ok=True)

        workflow_path = snakemake_path / (datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".Snakefile")
        with workflow_path.open("w") as outfile:
            outfile.write(workflow_string)

        return workflow_path

    def get_snakemake_path(self) -> Path:
        """Get the directory for storing files related to snakemake.

        Returns: the snakemake path as a string
        """
        return self.analysis_path / "snakemake"

    def save_figure(
        self,
        task: TaskOrName,
        figure: plt.Figure,
        figure_name: str,
        subdirectory: str = "figures",
        formats=[".png", ".pdf"],
    ) -> None:
        """Save the figure into the analysis results for this DataSet.

        This function will save the figure in both png and pdf formats.

        Args:
            task: the analysis task that generated this figure.
            figure: the figure handle for the figure to save
            figure_name: the name of the file to store the figure in, excluding
                    extension
            subdirectory: the name of the subdirectory within the specified
                    analysis task to save the figures.
            formats: formats to save figure as.
        """
        save_path = self.get_analysis_subdirectory(task, subdirectory) / figure_name

        if ".png" in formats:
            figure.savefig(save_path.with_suffix(".png"), pad_inches=0)
        if ".pdf" in formats:
            figure.savefig(save_path.with_suffix(".pdf"), transparent=True, pad_inches=0)

    def figure_exists(self, task: TaskOrName, figure_name: str, subdirectory: str = "figures") -> bool:
        """Determine if a figure with the specified name exists for the specified analysis task.

        This function only checks for the png formats.

        Args:
            task: the analysis task that generated this figure.
            figure_name: the name of the file to store the figure in, excluding
                    extension
            subdirectory: the name of the subdirectory within the specified
                    analysis task to save the figures.
        """
        save_path = self.get_analysis_subdirectory(task, subdirectory) / f"{figure_name}.png"
        return save_path.exists()

    def _analysis_result_save_path(
        self,
        resultName: str,
        analysisTask: TaskOrName,
        resultIndex: int = None,
        subdirectory: str = None,
        fileExtension: str = None,
    ) -> str:
        saveName = resultName
        if resultIndex is not None:
            saveName += "_" + str(resultIndex)
        if fileExtension is not None:
            saveName += fileExtension

        if analysisTask is None:
            return self.analysis_path / saveName
        else:
            return self.get_analysis_subdirectory(analysisTask, subdirectory) / saveName

    def list_analysis_files(
        self, analysisTask: TaskOrName = None, subdirectory: str = None, extension: str = None, fullPath: bool = True
    ) -> List[str]:
        basePath = self._analysis_result_save_path("", analysisTask, subdirectory=subdirectory)
        fileList = os.listdir(basePath)
        if extension:
            fileList = [x for x in fileList if x.endswith(extension)]
        if fullPath:
            fileList = [os.path.join(basePath, x) for x in fileList]
        return fileList

    def save_graph_as_gpickle(
        self,
        graph: nx.Graph,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ):
        """Save a networkx graph as a gpickle into the analysis results

        Args:
            graph: the networkx graph to save
            resultName: the base name of the output file
            analysisTask: the analysis task that the graph should be
                saved under. If None, the graph is saved to the
                data set root.
            resultIndex: index of the graph to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the graph
                should be saved to or None if the graph should be
                saved to the root directory for the analysis task.
        """
        savePath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".gpickle")
        nx.readwrite.gpickle.write_gpickle(graph, savePath)

    def load_graph_from_gpickle(
        self, resultName: str, analysisTask: TaskOrName = None, resultIndex: int = None, subdirectory: str = None
    ):
        """Load a networkx graph from a gpickle objective saved in the analysis
        results.

        Args:
            resultName: the base name of the output file
            analysisTask: the analysis task that the graph should be
                saved under. If None, the graph is saved to the
                data set root.
            resultIndex: index of the graph to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the graph
                should be saved to or None if the graph should be
                saved to the root directory for the analysis task.
        """
        savePath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".gpickle")
        return nx.readwrite.gpickle.read_gpickle(savePath)

    def save_dataframe_to_csv(
        self,
        dataframe: pandas.DataFrame,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
        **kwargs,
    ) -> None:
        """Save a pandas data frame to a csv file stored in this dataset.

        If a previous pandas data frame has been save with the same resultName,
        it will be overwritten

        Args:
            dataframe: the data frame to save
            resultName: the name of the output file
            analysisTask: the analysis task that the dataframe should be
                saved under. If None, the dataframe is saved to the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
            **kwargs: arguments to pass on to pandas.to_csv
        """
        savePath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".csv")

        with savePath.open("w") as f:
            dataframe.to_csv(f, **kwargs)

    def load_dataframe_from_csv(
        self,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
        **kwargs,
    ) -> pandas.DataFrame:
        """Load a pandas data frame from a csv file stored in this data set.

        Args:
            resultName:
            analysisTask:
            resultIndex:
            subdirectory:
            **kwargs:
        Returns:
            the pandas data frame
        Raises:
              FileNotFoundError: if the file does not exist
        """
        savePath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".csv")
        with savePath.open() as f:
            return pandas.read_csv(f, **kwargs)

    def open_pandas_hdfstore(
        self, mode: str, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ) -> pandas.HDFStore:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".h5")
        return pandas.HDFStore(savePath, mode=mode)

    def delete_pandas_hdfstore(
        self, resultName: str, analysisTask: TaskOrName = None, resultIndex: int = None, subdirectory: str = None
    ) -> None:
        hPath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".h5")
        if os.path.exists(hPath):
            os.remove(hPath)

    def open_table(
        self, mode: str, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ) -> tables.file:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".h5")
        return tables.open_file(savePath, mode=mode)

    def delete_table(
        self, resultName: str, analysisTask: TaskOrName = None, resultIndex: int = None, subdirectory: str = None
    ) -> None:
        """Delete an hdf5 file stored in this data set if it exists.

        Args:
            resultName: the name of the output file
            analysisTask: the analysis task that should be associated with this
                hdf5 file. If None, the file is assumed to be in the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
        """
        hPath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".h5")
        hPath.unlink(missing_ok=True)

    def open_hdf5_file(
        self,
        mode: str,
        resultName: str,
        analysisTask: TaskOrName = None,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> h5py.File:
        """Open an hdf5 file stored in this data set.

        Args:
            mode: the mode for opening the file, either 'r', 'r+', 'w', 'w-',
                or 'a'.
            resultName: the name of the output file
            analysisTask: the analysis task that should be associated with this
                hdf5 file. If None, the file is assumed to be in the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
        Returns:
            a h5py file object connected to the hdf5 file
        Raise:
            FileNotFoundError: if the mode is 'r' and the specified hdf5 file
                does not exist
        """
        hPath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".hdf5")
        if mode == "r" and not hPath.exists():
            raise FileNotFoundError(("Unable to open %s for reading since " + "it does not exist.") % hPath)

        return h5py.File(hPath, mode)

    def delete_hdf5_file(
        self, resultName: str, analysisTask: TaskOrName = None, resultIndex: int = None, subdirectory: str = None
    ) -> None:
        """Delete an hdf5 file stored in this data set if it exists.

        Args:
            resultName: the name of the output file
            analysisTask: the analysis task that should be associated with this
                hdf5 file. If None, the file is assumed to be in the
                data set root.
            resultIndex: index of the dataframe to save or None if no index
                should be specified
            subdirectory: subdirectory of the analysis task that the dataframe
                should be saved to or None if the dataframe should be
                saved to the root directory for the analysis task.
        """
        hPath = self._analysis_result_save_path(resultName, analysisTask, resultIndex, subdirectory, ".hdf5")
        hPath.unlink(missing_ok=True)

    def save_json_analysis_result(
        self,
        analysisResult: Dict,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".json")
        with savePath.open("w") as f:
            json.dump(analysisResult, f)

    def load_json_analysis_result(
        self, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ) -> Dict:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".json")
        with savePath.open() as f:
            return json.load(f)

    def load_pickle_analysis_result(
        self, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ) -> Dict:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".pkl")
        with savePath.open("rb") as f:
            return pickle.load(f)

    def save_pickle_analysis_result(
        self, analysisResult, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ):
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".pkl")
        with savePath.open("wb") as f:
            pickle.dump(analysisResult, f)

    def load_scanpy_analysis_result(
        self, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ) -> Dict:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".h5ad")
        return sc.read(savePath)

    def save_scanpy_analysis_result(
        self, analysisResult, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ):
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".h5ad")
        analysisResult.write(savePath)

    def save_numpy_analysis_result(
        self,
        analysisResult: np.ndarray,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory)
        np.save(savePath, analysisResult)

    def save_numpy_txt_analysis_result(
        self,
        analysisResult: np.ndarray,
        resultName: str,
        analysisName: str,
        resultIndex: int = None,
        subdirectory: str = None,
    ) -> None:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory)
        np.savetxt(savePath.with_suffix(savePath.suffix + ".csv"), analysisResult)

    def load_numpy_analysis_result(
        self, resultName: str, analysisName: str, resultIndex: int = None, subdirectory: str = None
    ) -> np.array:
        savePath = self._analysis_result_save_path(resultName, analysisName, resultIndex, subdirectory, ".npy")
        return np.load(savePath, allow_pickle=True)

    def load_numpy_analysis_result_if_available(
        self, resultName: str, analysisName: str, defaultValue, resultIndex: int = None, subdirectory: str = None
    ) -> np.array:
        """Load the specified analysis result or return the specified default
        value if the analysis result does not exist.

        Args:
            resultName: The name of the analysis result
            analysisName: The name of the analysis task the result is saved in
            defaultValue: The value to return if the specified analysis result
                does not exist
            resultIndex: The index of the analysi result
            subdirectory: The subdirectory within the analysis task that the
                result is saved in
        Returns: The analysis result or defaultValue if the analysis result
            doesn't exist.
        """
        try:
            return self.load_numpy_analysis_result(resultName, analysisName, resultIndex, subdirectory)
        except IOError:
            return defaultValue

    def get_analysis_subdirectory(self, analysisTask: TaskOrName, subdirectory: str = "", create: bool = True) -> Path:
        """
        analysisTask can either be the class or a string containing the
        class name.

        create - Flag indicating if the analysis subdirectory should be
            created if it does not already exist.
        """
        if isinstance(analysisTask, analysistask.AnalysisTask):
            analysisName = analysisTask.analysis_name
        else:
            analysisName = analysisTask

        if subdirectory:
            subdirectoryPath = self.analysis_path / analysisName / subdirectory
        else:
            subdirectoryPath = self.analysis_path / analysisName

        if create:
            subdirectoryPath.mkdir(parents=True, exist_ok=True)

        return subdirectoryPath

    def get_task_subdirectory(self, analysisTask: TaskOrName):
        return self.get_analysis_subdirectory(analysisTask, subdirectory="tasks")

    def get_log_subdirectory(self, analysisTask: TaskOrName):
        return self.get_analysis_subdirectory(analysisTask, subdirectory="log")

    def save_analysis_task(self, analysisTask: analysistask.AnalysisTask, overwrite: bool = False):
        saveName = self.get_task_subdirectory(analysisTask) / "task.json"

        try:
            existingTask = self.load_analysis_task(analysisTask.analysis_name, fragment="")

            existingParameters = existingTask.parameters.copy()
            existingVersion = existingParameters["merlin_version"]
            newParameters = analysisTask.parameters.copy()
            newVersion = newParameters["merlin_version"]

            if not merlin.is_compatible(existingVersion, newVersion):
                raise merlin.IncompatibleVersionError(
                    (
                        "Analysis task with name %s has been previously created "
                        + "with MERlin version %s, which is incompatible with "
                        + "the current MERlin version, %s. Please remove the "
                        + "old analysis folder to continue."
                    )
                    % (analysisTask.analysis_name, existingVersion, newVersion)
                )

            existingParameters.pop("merlin_version")
            newParameters.pop("merlin_version")

            # if not overwrite and not existingParameters == newParameters:
            #    print(existingParameters)
            #    print(newParameters)
            #    raise analysistask.AnalysisAlreadyExistsException(
            #        ('Analysis task with name %s already exists in this ' +
            #         'data set with different parameters.')
            #        % analysisTask.get_analysis_name())

        except FileNotFoundError:
            pass

        with saveName.open("w") as outFile:
            json.dump(analysisTask.parameters, outFile, indent=4)

    def load_analysis_task(self, task_name: str, fragment: str) -> analysistask.AnalysisTask:
        with Path(self.get_task_subdirectory(task_name), "task.json").open() as infile:
            parameters: dict[str, str] = json.load(infile)
            module = importlib.import_module(parameters["module"])
            task = getattr(module, parameters["class"])
            return task(self, self.analysis_path, parameters, task_name, fragment)

    def delete_analysis(self, analysisTask: TaskOrName) -> None:
        """
        Remove all files associated with the provided analysis
        from this data set.

        Before deleting an analysis task, it must be verified that the
        analysis task is not running.
        """
        analysisDirectory = self.get_analysis_subdirectory(analysisTask)
        shutil.rmtree(analysisDirectory)

    def get_analysis_tasks(self) -> List[str]:
        """
        Get a list of the analysis tasks within this dataset.

        Returns: A list of the analysis task names.
        """
        analysisList = []
        for a in os.listdir(self.analysis_path):
            if Path(self.analysis_path, a).is_dir() and Path(self.analysis_path, a, "tasks").exists():
                analysisList.append(a)
        analysisList.sort()
        return analysisList

    def analysis_exists(self, analysisTaskName: str) -> bool:
        """
        Determine if an analysis task with the specified name exists in this
        dataset.
        """
        analysisPath = self.get_analysis_subdirectory(analysisTaskName, create=False)
        return analysisPath.exists()

    def get_logger(self, task: analysistask.AnalysisTask, fragment: str = "") -> logging.Logger:
        logger_name = task.analysis_name
        if fragment:
            logger_name += "." + str(fragment)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self._log_path(task, fragment))
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def close_logger(self, task: analysistask.AnalysisTask, fragment: str = "") -> None:
        logger_name = task.analysis_name
        if fragment:
            logger_name += "." + str(fragment)

        logger = logging.getLogger(logger_name)

        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def _log_path(self, task: analysistask.AnalysisTask, fragment: str = "") -> str:
        log_name = task.analysis_name
        if fragment:
            log_name += "_" + str(fragment)
        log_name += ".log"

        return self.get_log_subdirectory(task) / log_name


class ImageDataSet(DataSet):
    def __init__(
        self,
        dataDirectoryName: Path,
        dataHome: Path | None = None,
        analysisHome: Path | None = None,
        microscopeParametersName: str | None = None,
        analysis_suffix: str | None = None,
    ):
        """Create a dataset for the specified raw data.

        Args:
            dataDirectoryName: the relative directory to the raw data
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
        """
        super().__init__(dataDirectoryName, dataHome, analysisHome, analysis_suffix)

        if microscopeParametersName is not None:
            self._import_microscope_parameters(microscopeParametersName)

        self._load_microscope_parameters()
        self.fovDimensions = [dim * self.micronsPerPixel for dim in self.imageDimensions]

    def get_image_file_names(self):
        return sorted(self.raw_data_portal.list_files(extensionList=[".dax", ".tif", ".tiff", ".zarr", ".zar"]))

    def load_image(self, imagePath, frameIndex):
        with imagereader.infer_reader(self.raw_data_portal.open_file(imagePath)) as reader:
            imageIn = reader.load_frame(int(frameIndex))
            if self.transpose:
                imageIn = np.transpose(imageIn)
            if self.flipHorizontal:
                imageIn = np.flip(imageIn, axis=1)
            if self.flipVertical:
                imageIn = np.flip(imageIn, axis=0)
            return imageIn

    def image_stack_size(self, imagePath):
        """
        Get the size of the image stack stored in the specified image path.

        Returns:
            a three element list with [width, height, frameCount] or None
                    if the file does not exist
        """
        with imagereader.infer_reader(self.raw_data_portal.open_file(imagePath)) as reader:
            return reader.film_size()

    def _import_microscope_parameters(self, microscopeParametersName):
        sourcePath = merlin.MICROSCOPE_PARAMETERS_HOME / microscopeParametersName
        destPath = self.analysis_path / "microscope_parameters.json"

        shutil.copyfile(sourcePath, destPath)

    def _load_microscope_parameters(self):
        path = self.analysis_path / "microscope_parameters.json"

        if os.path.exists(path):
            with open(path) as inputFile:
                self.microscopeParameters = json.load(inputFile)
        else:
            self.microscopeParameters = {}

        self.flipHorizontal = self.microscopeParameters.get("flip_horizontal", True)
        self.flipVertical = self.microscopeParameters.get("flip_vertical", False)
        self.transpose = self.microscopeParameters.get("transpose", True)
        self.micronsPerPixel = self.microscopeParameters.get("microns_per_pixel", 0.108)
        self.imageDimensions = self.microscopeParameters.get("image_dimensions", [2048, 2048])

    def get_microns_per_pixel(self):
        """Get the conversion factor to convert pixels to microns."""

        return self.micronsPerPixel

    def get_image_dimensions(self):
        """Get the dimensions of the images in this data set.

        Returns:
            A tuple containing the width and height of each image in pixels.
        """
        return self.imageDimensions

    def get_image_xml_metadata(self, imagePath: str) -> Dict:
        """Get the xml metadata stored for the specified image.

        Args:
            imagePath: the path to the image file (.dax or .tif)
        Returns: the metadata from the associated xml file
        """
        filePortal = self.raw_data_portal.open_file(imagePath).get_sibling_with_extension(".xml")
        return xmltodict.parse(filePortal.read_as_text())


Overlap = namedtuple("Overlap", ["fov", "xslice", "yslice"])


class MERFISHDataSet(ImageDataSet):
    def __init__(
        self,
        dataDirectoryName: Path,
        codebookNames: List[str] = None,
        dataOrganizationName: Path | None = None,
        positionFileName: str = None,
        dataHome: Path | None = None,
        analysisHome: Path | None = None,
        microscopeParametersName: str = None,
        fovList: str = None,
        skip: list = None,
        profile: bool = False,
        analysis_suffix: str | None = None,
    ):
        """Create a MERFISH dataset for the specified raw data.

        Args:
            dataDirectoryName: the relative directory to the raw data
            codebookNames: A list of the names of codebooks to use. The codebook
                    should be present in the analysis parameters
                    directory. Full paths can be provided for codebooks
                    present other directories.
            dataOrganizationName: the name of the data organization to use.
                    The data organization should be present in the analysis
                    parameters directory. A full path can be provided for
                    a codebook present in another directory.
            positionFileName: the name of the position file to use.
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
            fovList: a filename containing a list of FOV ids (one per line) that
                    MERlin will be run on. This can be used to process a subset of
                    the data. If not given, the entire dataset is processed.
        """
        super().__init__(dataDirectoryName, dataHome, analysisHome, microscopeParametersName, analysis_suffix)

        self.profile = profile
        self.dataOrganization = dataorganization.DataOrganization(self, dataOrganizationName, fovList, skip)
        if codebookNames:
            self.codebooks = [codebook.Codebook(self, name, i) for i, name in enumerate(codebookNames)]
        else:
            self.codebooks = self.load_codebooks()

        if positionFileName is not None:
            self._import_positions(positionFileName)
        self._load_positions()
        self._find_fov_overlaps()

    def save_codebook(self, codebook: codebook.Codebook) -> None:
        """Store the specified codebook in this dataset.

        If a codebook with the same codebook index and codebook name as the
        specified codebook already exists in this dataset, it is not
        overwritten.

        Args:
            codebook: the codebook to store
        Raises:
            FileExistsError: If a codebook with the same codebook index but
                a different codebook name is already save within this dataset.
        """
        existingCodebookName = self.get_stored_codebook_name(codebook.get_codebook_index())
        if existingCodebookName and existingCodebookName != codebook.get_codebook_name():
            raise FileExistsError(
                (
                    "Unable to save codebook %s with index %i "
                    + " since codebook %s already exists with "
                    + "the same index"
                )
                % (codebook.get_codebook_name(), codebook.get_codebook_index(), existingCodebookName)
            )

        if not existingCodebookName:
            self.save_dataframe_to_csv(
                codebook.get_data(),
                "_".join(["codebook", str(codebook.get_codebook_index()), codebook.get_codebook_name()]),
                index=False,
            )

    def load_codebooks(self) -> List[codebook.Codebook]:
        """Get all the codebooks stored within this dataset.

        Returns:
            A list of all the stored codebooks.
        """
        codebookList = []

        currentIndex = 0
        currentCodebook = self.load_codebook(currentIndex)
        while currentCodebook is not None:
            codebookList.append(currentCodebook)
            currentIndex += 1
            currentCodebook = self.load_codebook(currentIndex)

        return codebookList

    def load_codebook(self, codebookIndex: int = 0) -> Optional[codebook.Codebook]:
        """Load the codebook stored within this dataset with the specified
        index.

        Args:
            codebookIndex: the index of the codebook to load.
        Returns:
            The codebook stored with the specified codebook index. If no
            codebook exists with the specified index then None is returned.
        """
        codebookFile = [x for x in self.list_analysis_files(extension=".csv") if ("codebook_%i_" % codebookIndex) in x]
        if len(codebookFile) < 1:
            return None
        codebookName = "_".join(os.path.splitext(os.path.basename(codebookFile[0]))[0].split("_")[2:])
        return codebook.Codebook(self, codebookFile[0], codebookIndex, codebookName)

    def get_stored_codebook_name(self, codebookIndex: int = 0) -> Optional[str]:
        """Get the name of the codebook stored within this dataset with the
        specified index.

        Args:
            codebookIndex: the index of the codebook to load to find the name
                of.
        Returns:
            The name of the codebook stored with the specified codebook index.
            If no codebook exists with the specified index then None is
            returned.
        """
        codebookFile = [x for x in self.list_analysis_files(extension=".csv") if ("codebook_%i_" % codebookIndex) in x]
        if len(codebookFile) < 1:
            return None
        return "_".join(os.path.splitext(os.path.basename(codebookFile[0]))[0].split("_")[2:])

    def get_codebooks(self) -> List[codebook.Codebook]:
        """Get the codebooks associated with this dataset.

        Returns:
            A list containing the codebooks for this dataset.
        """
        return self.codebooks

    def get_codebook(self, codebookIndex: int = 0) -> codebook.Codebook:
        return self.codebooks[codebookIndex]

    def get_data_organization(self) -> dataorganization.DataOrganization:
        return self.dataOrganization

    def get_stage_positions(self) -> List[List[float]]:
        return self.positions

    def get_fov_offset(self, fov: int) -> Tuple[float, float]:
        """Get the offset of the specified fov in the global coordinate system.
        This offset is based on the anticipated stage position.

        Args:
            fov: index of the field of view
        Returns:
            A tuple specifying the x and y offset of the top right corner
            of the specified fov in pixels.
        """
        # TODO - this should be implemented using the position of the fov.
        return self.positions.loc[fov]["X"], self.positions.loc[fov]["Y"]

    def z_index_to_position(self, zIndex: int) -> float:
        """Get the z position associated with the provided z index."""

        return self.get_z_positions()[zIndex]

    def position_to_z_index(self, zPosition: float) -> int:
        """Get the z index associated with the specified z position

        Raises:
             Exception: If the provided z position is not specified in this
                dataset
        """

        zIndex = np.where(self.get_z_positions() == zPosition)[0]
        if len(zIndex) == 0:
            raise Exception("Requested z=%0.2f position not found." % zPosition)

        return zIndex[0]

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this dataset.

        Returns:
            A sorted list of all unique z positions
        """
        return self.dataOrganization.get_z_positions()

    def get_fovs(self) -> list[int]:
        return self.dataOrganization.get_fovs()

    def get_imaging_rounds(self) -> list[int]:
        # TODO - check this function
        return np.unique(self.dataOrganization.fileMap["imagingRound"])

    def get_raw_image(self, dataChannel, fov, zPosition=None):
        if zPosition is None:
            return np.array([self.get_raw_image(dataChannel, fov, zpos) for zpos in self.get_z_positions()])
        return self.load_image(
            self.dataOrganization.get_image_filename(dataChannel, fov),
            self.dataOrganization.get_image_frame_index(dataChannel, zPosition),
        )

    def get_fiducial_image(self, dataChannel, fov):
        index = self.dataOrganization.get_fiducial_frame_index(dataChannel)
        if isinstance(index, np.ndarray):
            return np.array(
                [self.load_image(self.dataOrganization.get_fiducial_filename(dataChannel, fov), i) for i in index]
            )
        return self.load_image(self.dataOrganization.get_fiducial_filename(dataChannel, fov), index)

    def _import_positions_from_metadata(self):
        positionData = []
        for f in self.get_fovs():
            metadata = self.get_image_xml_metadata(self.dataOrganization.get_image_filename(0, f))
            currentPositions = metadata["settings"]["acquisition"]["stage_position"]["#text"].split(",")
            positionData.append([float(x) for x in currentPositions])
        positionPath = self.analysis_path / "positions.csv"
        np.savetxt(positionPath, np.array(positionData), delimiter=",")

    def _load_positions(self):
        positionPath = self.analysis_path / "positions.csv"
        if not positionPath.exists():
            self._import_positions_from_metadata()
        self.positions = pandas.read_csv(positionPath, header=None, names=["X", "Y"])
        self.positions.index = self.get_fovs()

    def _import_positions(self, positionFileName):
        sourcePath = merlin.POSITION_HOME / positionFileName
        destPath = self.analysis_path / "positions.csv"

        shutil.copyfile(sourcePath, destPath)

    def _convert_parameter_list(self, listIn, castFunction, delimiter=";"):
        return [castFunction(x) for x in listIn.split(delimiter) if len(x) > 0]

    def _get_overlap_slice(self, diff: float, axis: int, get_trim: bool = False) -> slice:
        """Get a slice for the region of an image overlapped by another FOV.
        :param diff: The amount of overlap in the global coordinate system.
        :param axis: The axis for the slice.
        :param get_trim: If True, return the half of the overlap closest to the edge. This is for
            determining in which region the barcodes should be trimmed to avoid duplicates.
        :return: A slice in the FOV coordinate system for the overlap.
        """
        fovsize = self.fovDimensions[axis]
        imagesize = self.imageDimensions[axis]
        if int(diff) == 0:
            return slice(None)
        if diff > 0:
            if get_trim:
                diff = fovsize - ((fovsize - diff) / 2)
            overlap = imagesize * diff / fovsize
            return slice(math.trunc(overlap), None)
        else:
            if get_trim:
                diff = -fovsize - ((-fovsize - diff) / 2)
            overlap = imagesize * diff / fovsize
            return slice(None, math.trunc(overlap))

    def _find_fov_overlaps(self) -> None:
        positions = self.get_stage_positions()
        neighbor_graph = NearestNeighbors()
        neighbor_graph = neighbor_graph.fit(positions)
        radius = max(self.fovDimensions)
        res = neighbor_graph.radius_neighbors(positions, radius=radius, return_distance=True, sort_results=True)
        self.overlaps = {}
        self.trim_overlaps = {}
        for i, (dists, fovs) in enumerate(zip(*res)):
            i = positions.iloc[i].name
            for dist, fov in zip(dists, fovs):
                fov = positions.iloc[fov].name
                if int(dist) == 0 or f"{i}__{fov}" in self.overlaps or f"{fov}__{i}" in self.overlaps:
                    continue
                diff = positions.loc[i] - positions.loc[fov]
                if self.flipVertical:
                    diff["X"] = -diff["X"]
                if not self.flipHorizontal:
                    diff["Y"] = -diff["Y"]
                _get_x_slice = functools.partial(self._get_overlap_slice, axis=0)
                _get_y_slice = functools.partial(self._get_overlap_slice, axis=1)
                self.overlaps[f"{i}__{fov}"] = (
                    Overlap(i, _get_x_slice(diff["X"], get_trim=False), _get_y_slice(-diff["Y"], get_trim=False)),
                    Overlap(fov, _get_x_slice(-diff["X"], get_trim=False), _get_y_slice(diff["Y"], get_trim=False)),
                )
                self.trim_overlaps[f"{i}__{fov}"] = (
                    Overlap(i, _get_x_slice(diff["X"], get_trim=True), _get_y_slice(-diff["Y"], get_trim=True)),
                    Overlap(fov, _get_x_slice(-diff["X"], get_trim=True), _get_y_slice(diff["Y"], get_trim=True)),
                )

    def get_overlap(self, overlap_name: str) -> tuple[Overlap]:
        return self.overlaps[overlap_name]

    def get_overlap_names(self) -> list[str]:
        """Get the names of all overlaps between FOVs.

        Each name is a pair of FOV ids separate by a double underscore.
        """
        return list(self.overlaps.keys())

    def get_overlap_mask(self, fov: str, *, trim: bool = False) -> np.ndarray:
        """Get an image mask indicating the regions of the FOV overlapped by other FOVs."""
        mask = np.zeros(self.imageDimensions, dtype=np.uint16)
        overlaps = self.trim_overlaps if trim else self.overlaps
        overlap_names = filter(lambda name: fov in name.split("__"), overlaps)
        for name in overlap_names:
            i = name.split("__").index(fov)
            mask[overlaps[name][i].xslice, overlaps[name][i].yslice] = 1
        return mask
