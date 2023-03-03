import pandas
import numpy as np
import pandas as pd

from merlin.core import analysistask


class PartitionBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters["filter_task"], self.parameters["assignment_task"], self.parameters["alignment_task"]]

    def get_partitioned_barcodes(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pandas.concat([self.get_partitioned_barcodes(fov) for fov in self.dataSet.get_fovs()])

        return self.dataSet.load_dataframe_from_csv("counts_per_cell", self.get_analysis_name(), fov, index_col=0)

    def _run_analysis(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(self.parameters["filter_task"])
        assignmentTask = self.dataSet.load_analysis_task(self.parameters["assignment_task"])
        alignTask = self.dataSet.load_analysis_task(self.parameters["alignment_task"])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()
        for fi in fovIntersections:
            partialBC = bcDB.get_barcodes(fi)
            if fi == fovIntersections[0]:
                currentFOVBarcodes = partialBC.copy(deep=True)
            else:
                currentFOVBarcodes = pandas.concat([currentFOVBarcodes, partialBC], 0)

        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        countsDF = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells],
        )

        for cell in currentCells:
            contained = cell.contains_positions(currentFOVBarcodes.loc[:, ["global_x", "global_y", "z"]].values)
            count = currentFOVBarcodes[contained].groupby("barcode_id").size()
            count = count.reindex(range(barcodeCount), fill_value=0)
            countsDF.loc[cell.get_feature_id(), :] = count.values.tolist()

        barcodeNames = [codebook.get_name_for_barcode_index(x) for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(countsDF, "counts_per_cell", self.get_analysis_name(), fragmentIndex)


class ExportPartitionedBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that combines counts per cells data from each
    field of view into a single output file.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters["partition_task"]]

    def _run_analysis(self):
        pTask = self.dataSet.load_analysis_task(self.parameters["partition_task"])
        parsedBarcodes = pTask.get_partitioned_barcodes()

        self.dataSet.save_dataframe_to_csv(parsedBarcodes, "barcodes_per_feature", self.get_analysis_name())


class PartitionBarcodesFromMask(analysistask.ParallelAnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on segmentation masks produced during the segment task.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters["segment_task"], self.parameters["filter_task"]]

    def get_cell_by_gene_matrix(self, fov: str = None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pandas.concat([self.get_cell_by_gene_matrix(fov) for fov in self.dataSet.get_fovs()])

        return self.dataSet.load_dataframe_from_csv(
            "counts_per_cell", self.get_analysis_name(), fov, subdirectory="counts_per_cell", index_col=0
        )

    def get_barcode_table(self, fov=None):
        if fov is None:
            return pandas.concat([self.get_barcode_table(fov) for fov in self.dataSet.get_fovs()])

        return self.dataSet.load_dataframe_from_csv(
            "barcodes", self.get_analysis_name(), fov, subdirectory="barcodes", index_col=0
        )

    def apply_mask(self, barcodes, mask):
        return mask[barcodes["x"].round().astype(int), barcodes["y"].round().astype(int)]

    def _run_analysis(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(self.parameters["filter_task"])
        segmentTask = self.dataSet.load_analysis_task(self.parameters["segment_task"])
        codebook = filterTask.get_codebook()
        barcodes = filterTask.get_barcode_database().get_barcodes(fragmentIndex)

        # Trim barcodes in overlapping regions
        overlap_mask = self.dataSet.get_overlap_mask(fragmentIndex, trim=True)
        barcodes = barcodes[~self.apply_mask(barcodes, overlap_mask.astype(bool))]

        cell_mask = segmentTask.load_mask(fragmentIndex)
        barcodes["cell_id"] = self.apply_mask(barcodes, cell_mask).astype(str)
        barcodes["cell_id"] = fragmentIndex + "__" + barcodes["cell_id"]

        # Save barcode table
        barcodes["gene"] = [codebook.get_name_for_barcode_index(i) for i in barcodes["barcode_id"]]
        barcodes = barcodes[["gene", "cell_id", "fov", "x", "y", "z", "global_x", "global_y", "global_z"]]
        self.dataSet.save_dataframe_to_csv(
            barcodes, "barcodes", self.get_analysis_name(), fragmentIndex, subdirectory="barcodes", index=False
        )

        # Make cell by gene matrix
        matrix = pd.crosstab(barcodes["cell_id"], barcodes["gene"]).drop(fragmentIndex + "__0")
        matrix.columns.name = None
        matrix.index.name = None
        self.dataSet.save_dataframe_to_csv(
            matrix, "counts_per_cell", self.get_analysis_name(), fragmentIndex, subdirectory="counts_per_cell"
        )
