import pandas as pd
import numpy as np
import scanpy as sc

from merlin.core import analysistask


class FinalOutput(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters["partition_task"], self.parameters["cell_metadata_task"]]

    def _run_analysis(self):
        metadataTask = self.dataSet.load_analysis_task(self.parameters["cell_metadata_task"])
        partitionTask = self.dataSet.load_analysis_task(self.parameters["partition_task"])
        cell_mapping = metadataTask.get_cell_mapping()

        barcodes = partitionTask.get_barcode_table()
        barcodes["cell_id"] = [
            cell_mapping[cell_id] if cell_id in cell_mapping else cell_id for cell_id in barcodes["cell_id"]
        ]
        self.dataSet.save_dataframe_to_csv(barcodes, "detected_transcripts", self, index=False)

        matrix = partitionTask.get_cell_by_gene_matrix()
        matrix.index = [cell_mapping[cell_id] if cell_id in cell_mapping else cell_id for cell_id in matrix.index]
        matrix = matrix.reset_index().groupby("index").sum()
        matrix.index.name = None
        self.dataSet.save_dataframe_to_csv(matrix, "cell_by_gene", self)

        celldata = metadataTask.get_cell_metadata()
        blank_cols = np.array(["notarget" in col or "blank" in col.lower() for col in matrix])
        adata = sc.AnnData(matrix.loc[:, ~blank_cols], dtype=np.uint32)
        adata.obsm["X_blanks"] = matrix.loc[:, blank_cols].to_numpy()
        adata.uns["blank_names"] = matrix.columns[blank_cols].to_list()
        adata.obsm["X_spatial"] = np.array(
            celldata[["global_x", "global_y"]].reindex(index=adata.obs.index)
        )
        adata.obs["volume"] = celldata["volume"]
        adata.obs["fov"] = [cell_id.split("__")[0] for cell_id in adata.obs.index]
        adata.layers["counts"] = adata.X
        self.dataSet.save_scanpy_analysis_result(adata, "scanpy_object", self)
