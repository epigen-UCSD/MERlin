import pandas as pd
import numpy as np
import scanpy as sc

from merlin.core import analysistask


class FinalOutput(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(self.parameters["segment_task"])
        self.linkCellTask = self.dataSet.load_analysis_task(self.parameters["link_cell_task"])

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters["partition_task"], self.parameters["segment_task"], self.parameters["link_cell_task"]]

    def _combine_overlap_volumes(self):
        volumes = pd.concat(
            [self.linkCellTask.get_overlap_volumes(overlapName) for overlapName in self.dataSet.get_overlap_names()]
        )
        return volumes.groupby("label").max()

    def get_scanpy_object(self):
        return self.dataSet.load_scanpy_analysis_result("scanpy_object", self)

    def get_cell_metadata_table(self):
        try:
            return self.dataSet.load_dataframe_from_csv("cell_metadata", self.get_analysis_name(), index_col=0)
        except FileNotFoundError:
            dfs = []
            cell_mapping = self.linkCellTask.get_cell_mapping()
            for fov in self.dataSet.get_fovs():
                df = self.segmentTask.load_metadata(fov)
                df["cell_id"] = fov + "__" + df["cell_id"].astype(str)
                df = df.rename(columns={"volume": "fov_volume"})
                dfs.append(df)
            metadata = pd.concat(dfs).set_index("cell_id")
            metadata["overlap_volume"] = self._combine_overlap_volumes()
            metadata["overlap_volume"] = metadata["overlap_volume"].fillna(0)
            metadata["nonoverlap_volume"] = metadata["fov_volume"] - metadata["overlap_volume"]
            metadata.index = [
                cell_mapping[cell_id] if cell_id in cell_mapping else cell_id for cell_id in metadata.index
            ]
            metadata.index.name = "cell_id"
            metadata = metadata.groupby("cell_id").agg(
                {"global_x": "mean", "global_y": "mean", "overlap_volume": "mean", "nonoverlap_volume": "sum"}
            )
            metadata["volume"] = metadata["overlap_volume"] + metadata["nonoverlap_volume"]
            metadata = metadata.drop(columns=["overlap_volume", "nonoverlap_volume"])
            self.dataSet.save_dataframe_to_csv(metadata, "cell_metadata", self.get_analysis_name())
            return metadata

    def _run_analysis(self):
        partitionTask = self.dataSet.load_analysis_task(self.parameters["partition_task"])
        cell_mapping = self.linkCellTask.get_cell_mapping()

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

        celldata = self.get_cell_metadata_table()
        blank_cols = np.array(["notarget" in col or "blank" in col.lower() for col in matrix])
        adata = sc.AnnData(matrix.loc[:, ~blank_cols], dtype=np.uint32)
        adata.obsm["X_blanks"] = matrix.loc[:, blank_cols].to_numpy()
        adata.uns["blank_names"] = matrix.columns[blank_cols].to_list()
        adata.obsm["X_spatial"] = np.array(celldata[["global_x", "global_y"]].reindex(index=adata.obs.index))
        adata.obs["volume"] = celldata["volume"]
        adata.obs["fov"] = [cell_id.split("__")[0] for cell_id in adata.obs.index]
        adata.layers["counts"] = adata.X
        sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata, base=2)
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X", metric="cosine")
        sc.tl.leiden(adata)
        sc.tl.umap(adata, min_dist=0.3)
        self.dataSet.save_scanpy_analysis_result(adata, "scanpy_object", self)
