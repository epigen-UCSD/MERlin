import numpy as np
import pandas as pd

from merlin.core import analysistask


class PartitionBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"filter_task": [], "assignment_task": [], "alignment_task": []})

        self.define_results(("counts_per_cell", {"index": True}))

    def get_partitioned_barcodes(self, fov: int = None) -> pd.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pd.concat([self.get_partitioned_barcodes(fov) for fov in self.dataSet.get_fovs()])

        return self.dataSet.load_dataframe_from_csv("counts_per_cell", self.analysis_name, fov, index_col=0)

    def run_analysis(self, fragmentIndex):
        fovBoxes = self.alignment_task.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if fovBoxes[fragmentIndex].intersects(x)])

        codebook = self.filter_task.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = self.filter_task.get_barcode_database()
        for fi in fovIntersections:
            partialBC = bcDB.get_barcodes(fi)
            if fi == fovIntersections[0]:
                currentFOVBarcodes = partialBC.copy(deep=True)
            else:
                currentFOVBarcodes = pd.concat([currentFOVBarcodes, partialBC], 0)

        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)

        sDB = self.assignment_task.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        countsDF = pd.DataFrame(
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
        self.counts_per_cell = countsDF


class ExportPartitionedBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that combines counts per cells data from each
    field of view into a single output file.
    """

    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies({"partition_task": []})

        self.define_results("barcodes_per_feature")

    def run_analysis(self):
        self.barcodes_per_feature = self.partition_task.get_partitioned_barcodes()


class PartitionBarcodesFromMask(analysistask.AnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on segmentation masks produced during the segment task.
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"segment_task": ["mask"], "filter_task": []})

        self.define_results(("barcodes", {"index": False}), "counts_per_cell")

    def get_cell_by_gene_matrix(self, fov: str = None) -> pd.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pd.concat([self.get_cell_by_gene_matrix(fov) for fov in self.dataSet.get_fovs()]).fillna(0)

        return self.dataSet.load_dataframe_from_csv(
            "counts_per_cell", self.analysis_name, fov, subdirectory="counts_per_cell", index_col=0
        )

    def get_barcode_table(self, fov=None):
        if fov is None:
            return pd.concat([self.get_barcode_table(fov) for fov in self.dataSet.get_fovs()])

        return self.dataSet.load_dataframe_from_csv(
            "barcodes", self.analysis_name, fov, subdirectory="barcodes"
        )

    def apply_mask(self, barcodes, mask):
        if mask.ndim == 2:
            return mask[barcodes["x"].round().astype(int), barcodes["y"].round().astype(int)]
        return mask[
            barcodes["z"].round().astype(int), barcodes["x"].round().astype(int), barcodes["y"].round().astype(int)
        ]

    def assign_barcodes(self, filter_task):
        codebook = filter_task.get_codebook()
        barcodes = filter_task.load_result("barcodes", self.fragment)

        barcodes = pd.DataFrame(
            barcodes[:, [-1, 5, 6, 7, 8, 9, 10]],
            columns=["barcode_id", "x", "y", "z", "global_x", "global_y", "global_z"],
        )
        barcodes["fov"] = self.fragment
        # Trim barcodes in overlapping regions
        overlap_mask = self.dataSet.get_overlap_mask(self.fragment, trim=True)
        barcodes = barcodes[~self.apply_mask(barcodes, overlap_mask.astype(bool))]

        cell_mask = self.segment_task.load_mask()
        barcodes["cell_id"] = self.apply_mask(barcodes, cell_mask).astype(str)
        barcodes["cell_id"] = self.fragment + "__" + barcodes["cell_id"]

        barcodes["gene"] = [codebook.get_name_for_barcode_index(i) for i in barcodes["barcode_id"]]

        return barcodes[["gene", "cell_id", "fov", "x", "y", "z", "global_x", "global_y", "global_z"]]

    def run_analysis(self) -> None:
        if isinstance(self.filter_task, list):
            self.barcodes = pd.concat([self.assign_barcodes(task) for task in self.filter_task])
        else:
            self.barcodes = self.assign_barcodes(self.filter_task)

        # Make cell by gene matrix
        matrix = pd.crosstab(self.barcodes["cell_id"], self.barcodes["gene"])
        try:
            matrix = matrix.drop(self.fragment + "__0")
        except KeyError:
            pass
        matrix.columns.name = None
        matrix.index.name = None
        self.counts_per_cell = matrix

    def metadata(self) -> dict:
        if len(self.barcodes) == 0:
            return {"fraction": 0}
        else:
            return {
                "fraction": len(self.barcodes[~self.barcodes["cell_id"].str.endswith("__0")]) / len(self.barcodes)
            }
