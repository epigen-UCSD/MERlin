import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from merlin.analysis import filterbarcodes
from merlin.plots._base import AbstractPlot, PlotMetadata


class MinimumDistanceDistributionPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"decode_task": "all"})
        self.set_required_metadata(DecodedBarcodesMetadata)

    def create_plot(self, **kwargs):
        metadata = kwargs["metadata"]["decodeplots/DecodedBarcodesMetadata"]

        distance = metadata.distance_bins[:-1]
        shift = (distance[0] + distance[1]) / 2
        distance = [x + shift for x in distance]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(distance, metadata.distance_counts)
        plt.xlabel("Barcode distance")
        plt.ylabel("Count")
        plt.title("Distance distribution for all barcodes")

        return fig


class AreaDistributionPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"decode_task": "all"})
        self.set_required_metadata(DecodedBarcodesMetadata)

    def create_plot(self, **kwargs):
        metadata = kwargs["metadata"]["decodeplots/DecodedBarcodesMetadata"]
        area = metadata.area_bins[:-1]
        shift = (area[0] + area[1]) / 2
        area = [x + shift for x in area]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(area, metadata.area_counts, width=2 * shift)
        plt.xlabel("Barcode area (pixels)")
        plt.ylabel("Count")
        plt.title("Area distribution for all barcodes")

        return fig


class MeanIntensityDistributionPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"decode_task": "all"})
        self.set_required_metadata(DecodedBarcodesMetadata)

    def create_plot(self, **kwargs):
        metadata = kwargs["metadata"]["decodeplots/DecodedBarcodesMetadata"]
        intensity = metadata.intensity_bins[:-1]
        shift = (intensity[0] + intensity[1]) / 2
        intensity = [x + shift for x in intensity]

        fig = plt.figure(figsize=(4, 4))
        plt.bar(intensity, metadata.intensity_counts, width=2 * shift)
        plt.xlabel("Mean intensity ($log_{10}$)")
        plt.ylabel("Count")
        plt.title("Intensity distribution for all barcodes")

        return fig


class DecodedBarcodeAbundancePlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"decode_task": "all"})
        self.set_required_metadata(DecodedBarcodesMetadata)

    def create_plot(self, **kwargs):
        decode_task = kwargs["tasks"]["decode_task"]
        codebook = decode_task.get_codebook()
        metadata = kwargs["metadata"]["decodeplots/DecodedBarcodesMetadata"]

        counts = pd.DataFrame(
            metadata.barcode_counts, index=np.arange(len(metadata.barcode_counts)), columns=["counts"]
        )

        gene_counts = counts[counts.index.isin(codebook.get_coding_indexes())].sort_values(
            by="counts", ascending=False
        )
        blank_counts = counts[counts.index.isin(codebook.get_blank_indexes())].sort_values(
            by="counts", ascending=False
        )

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(gene_counts)), np.log10(gene_counts["counts"]), "b.")
        plt.plot(np.arange(len(gene_counts), len(counts)), np.log10(blank_counts["counts"]), "r.")
        plt.xlabel("Sorted barcode index")
        plt.ylabel("Count (log10)")
        plt.title("Barcode abundances")
        plt.legend(["Coding", "Blank"])

        return fig


class AreaIntensityViolinPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"decode_task": "all"})
        self.set_required_metadata(DecodedBarcodesMetadata)

    def create_plot(self, **kwargs):
        metadata = kwargs["metadata"]["decodeplots/DecodedBarcodesMetadata"]
        tasks = kwargs["tasks"]

        bins = metadata.intensity_bins[:-1]

        def distribution_dict(count_list):
            if np.max(count_list) > 0:
                return {
                    "coords": bins,
                    "vals": count_list,
                    "mean": np.average(bins, weights=count_list),
                    "median": np.average(bins, weights=count_list),
                    "min": bins[np.nonzero(count_list)[0]],
                    "max": bins[np.nonzero(count_list)[-1]],
                }
            return {"coords": bins, "vals": count_list, "mean": 0, "median": 0, "min": 0, "max": 0}

        vpstats = [distribution_dict(x) for x in metadata.intensity_by_area]

        fig = plt.figure(figsize=(15, 5))
        ax = plt.subplot(1, 1, 1)
        ax.violin(vpstats, positions=metadata.area_bins[:-1], showmeans=True, showextrema=False)

        if "filter_task" in tasks and isinstance(tasks["filter_task"], filterbarcodes.FilterBarcodes):
            plt.axvline(x=tasks["filter_task"].parameters["area_threshold"] - 0.5, color="green", linestyle=":")
            plt.axhline(
                y=np.log10(tasks["filter_task"].parameters["intensity_threshold"]), color="green", linestyle=":"
            )

        plt.xlabel("Barcode area (pixels)")
        plt.ylabel("Mean intensity ($log_{10}$)")
        plt.title("Intensity distribution by barcode area")
        plt.xlim([0, 17])

        return fig


class DecodedBarcodesMetadata(PlotMetadata):
    def __init__(self, plot_task, required_tasks):
        super().__init__(plot_task, required_tasks)

        self.decode_task = self.required_tasks["decode_task"]
        self.queued_data = []

        self.area_bins = np.arange(25)

        self.register_updaters({"decode_task": self.process_barcodes})
        self.register_datasets(
            "barcode_counts",
            "area_counts",
            "intensity_bins",
            "intensity_counts",
            "distance_bins",
            "distance_counts",
            "intensity_by_area",
        )

    def _determine_bins(self) -> None:
        barcode_data = pd.concat(self.queued_data)
        min_intensity = np.log10(barcode_data["mean_intensity"].min())
        max_intensity = np.log10(barcode_data["mean_intensity"].max())
        self.intensity_bins = np.linspace(min_intensity, max_intensity, 100)

        min_distance = barcode_data["min_distance"].min()
        max_distance = barcode_data["min_distance"].max()
        self.distance_bins = np.linspace(min_distance, max_distance, 100)

        codebook = self.decode_task.get_codebook()
        self.barcode_counts = np.zeros(codebook.get_barcode_count())
        self.area_counts = np.zeros(len(self.area_bins) - 1)
        self.intensity_by_area = np.zeros((len(self.area_bins) - 1, len(self.intensity_bins) - 1))
        self.distance_counts = np.zeros(len(self.distance_bins) - 1)
        self.intensity_counts = np.zeros(len(self.intensity_bins) - 1)

    def _extract_from_barcodes(self, barcodes) -> None:
        self.barcode_counts += np.histogram(barcodes["barcode_id"], bins=np.arange(len(self.barcode_counts) + 1))[0]
        self.area_counts += np.histogram(barcodes["area"], bins=self.area_bins)[0]
        self.intensity_counts += np.histogram(np.log10(barcodes["mean_intensity"]), bins=self.intensity_bins)[0]
        self.distance_counts += np.histogram(np.log10(barcodes["min_distance"]), bins=self.distance_bins)[0]

        for i, current_area in enumerate(self.area_bins[:-1]):
            self.intensity_by_area[i, :] += np.histogram(
                np.log10(barcodes[barcodes["area"] == current_area]["mean_intensity"]), bins=self.intensity_bins
            )[0]

    def process_barcodes(self, fov) -> None:
        barcodes = self.decode_task.get_barcode_database().get_barcodes(
            fov, columnList=["barcode_id", "area", "mean_intensity", "min_distance"]
        )
        if not hasattr(self, "intensity_bins"):  # Still waiting for 20 fovs to complete
            self.queued_data.append(barcodes)
            if len(self.queued_data) >= min(20, len(self.decode_task.fragment_list) - 1):
                self._determine_bins()
                for barcode_data in self.queued_data:
                    self._extract_from_barcodes(barcode_data)
                self.queued_data = []
        else:
            self._extract_from_barcodes(barcodes)
