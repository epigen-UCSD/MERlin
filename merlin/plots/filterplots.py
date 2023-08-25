from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from merlin.plots._base import AbstractPlot, PlotMetadata


class CodingBarcodeSpatialDistribution(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all", "global_align_task": "all"})
        self.set_required_metadata(GlobalSpatialDistributionMetadata)

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        metadata = kwargs["metadata"]["filterplots/GlobalSpatialDistributionMetadata"]
        plt.imshow(
            metadata.spatial_coding_counts,
            extent=metadata.get_spatial_extents(),
            cmap=plt.get_cmap("Greys"),
        )
        plt.xlabel("X position (pixels)")
        plt.ylabel("Y position (pixels)")
        plt.title("Spatial distribution of coding barcodes")
        cbar = plt.colorbar(ax=ax)
        cbar.set_label("Barcode count", rotation=270)

        return fig


class BlankBarcodeSpatialDistribution(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all", "global_align_task": "all"})
        self.set_required_metadata(GlobalSpatialDistributionMetadata)

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        metadata = kwargs["metadata"]["filterplots/GlobalSpatialDistributionMetadata"]
        plt.imshow(
            metadata.spatial_blank_counts,
            extent=metadata.get_spatial_extents(),
            cmap=plt.get_cmap("Greys"),
        )
        plt.xlabel("X position (pixels)")
        plt.ylabel("Y position (pixels)")
        plt.title("Spatial distribution of blank barcodes")
        cbar = plt.colorbar(ax=ax)
        cbar.set_label("Barcode count", rotation=270)

        return fig


class BarcodeRadialDensityPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all"})
        self.set_required_metadata(FOVSpatialDistributionMetadata)

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(7, 7))

        metadata = kwargs["metadata"]["filterplots/FOVSpatialDistributionMetadata"]
        single_counts = metadata.single_color_counts
        plt.plot(metadata.radial_bins[:-1], single_counts / np.sum(single_counts))
        multi_counts = metadata.multi_color_counts
        plt.plot(metadata.radial_bins[:-1], multi_counts / np.sum(multi_counts))
        plt.legend(["Single color barcodes", "Multi color barcodes"])
        plt.xlabel("Radius (pixels)")
        plt.ylabel("Normalized radial barcode density")

        return fig


class CodingBarcodeFOVDistributionPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all"})
        self.set_required_metadata(FOVSpatialDistributionMetadata)

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        metadata = kwargs["metadata"]["filterplots/FOVSpatialDistributionMetadata"]
        plt.imshow(
            metadata.spatial_coding_counts,
            extent=metadata.get_spatial_extents(),
            cmap=plt.get_cmap("Greys"),
        )
        plt.xlabel("X position (pixels)")
        plt.ylabel("Y position (pixels)")
        plt.title("Spatial distribution of coding barcodes within FOV")
        cbar = plt.colorbar(ax=ax)
        cbar.set_label("Barcode count", rotation=270)

        return fig


class BlankBarcodeFOVDistributionPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all"})
        self.set_required_metadata(FOVSpatialDistributionMetadata)

    def create_plot(self, **kwargs):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        metadata = kwargs["metadata"]["filterplots/FOVSpatialDistributionMetadata"]
        plt.imshow(
            metadata.spatial_blank_counts,
            extent=metadata.get_spatial_extents(),
            cmap=plt.get_cmap("Greys"),
        )
        plt.xlabel("X position (pixels)")
        plt.ylabel("Y position (pixels)")
        plt.title("Spatial distribution of blank barcodes within FOV")
        cbar = plt.colorbar(ax=ax)
        cbar.set_label("Barcode count", rotation=270)

        return fig


class FilteredBarcodeAbundancePlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all"})
        self.set_required_metadata(FilteredBarcodesMetadata)

    def create_plot(self, **kwargs):
        filter_task = kwargs["tasks"]["filter_task"]
        codebook = filter_task.get_codebook()
        metadata = kwargs["metadata"]["filterplots/FilteredBarcodesMetadata"]

        counts = pd.DataFrame(
            metadata.barcode_counts, index=np.arange(len(metadata.barcode_counts)), columns=["counts"]
        )

        gene_counts = counts[counts.index.isin(codebook.get_coding_indexes())].sort_values(by="counts", ascending=False)
        blank_counts = counts[counts.index.isin(codebook.get_blank_indexes())].sort_values(by="counts", ascending=False)

        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(gene_counts)), np.log10(gene_counts["counts"]), "b.")
        plt.plot(np.arange(len(gene_counts), len(counts)), np.log10(blank_counts["counts"]), "r.")
        plt.xlabel("Sorted barcode index")
        plt.ylabel("Count (log10)")
        plt.title("Barcode abundances")
        plt.legend(["Coding", "Blank"])
        plt.tight_layout(pad=0.2)

        return fig


class AdaptiveFilterBarcodeDistributionPlots(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all"})

    def create_plot(self, **kwargs):
        filter_task = kwargs["tasks"]["filter_task"]
        adaptive_task = filter_task.get_adaptive_thresholds()
        blank_histogram = adaptive_task.get_blank_count_histogram()
        coding_histogram = adaptive_task.get_coding_count_histogram()
        blank_fraction = adaptive_task.get_blank_fraction_histogram()
        threshold = adaptive_task.calculate_threshold_for_misidentification_rate(
            filter_task.parameters["misidentification_rate"]
        )
        area_bins = adaptive_task.get_area_bins()
        intensity_bins = adaptive_task.get_intensity_bins()
        distance_bins = adaptive_task.get_distance_bins()
        plot_extent = (distance_bins[0], distance_bins[-1], intensity_bins[0], intensity_bins[-1])

        fig = plt.figure(figsize=(20, 30))
        for i in range(min(len(area_bins), 6)):
            plt.subplot(6, 4, 4 * i + 1)
            plt.imshow(
                blank_histogram[:, :, i].T + coding_histogram[:, :, i].T,
                extent=plot_extent,
                origin="lower",
                aspect="auto",
                cmap="OrRd",
            )
            cbar = plt.colorbar()
            cbar.set_label("Barcode count", rotation=270, labelpad=8)
            plt.ylabel(f"Area={area_bins[i]}\nMean intensity (log10)")
            plt.xlabel("Minimum distance")
            if i == 0:
                plt.title("Distribution of all barcodes")

            plt.subplot(6, 4, 4 * i + 2)
            plt.imshow(blank_histogram[:, :, i].T, extent=plot_extent, origin="lower", aspect="auto", cmap="OrRd")
            cbar = plt.colorbar()
            cbar.set_label("Blank count", rotation=270, labelpad=8)
            plt.ylabel("Mean intensity (log10)")
            plt.xlabel("Minimum distance")
            if i == 0:
                plt.title("Distribution of blank barcodes")

            plt.subplot(6, 4, 4 * i + 3)
            plt.imshow(
                blank_fraction[:, :, i].T, extent=plot_extent, origin="lower", aspect="auto", cmap="OrRd", vmax=1.0
            )
            cbar = plt.colorbar()
            cbar.set_label("Blank fraction", rotation=270, labelpad=8)
            plt.ylabel("Mean intensity (log10)")
            plt.xlabel("Minimum distance")
            if i == 0:
                plt.title("Distribution of normalized blank fraction")

            plt.subplot(6, 4, 4 * i + 4)
            plt.imshow(
                blank_fraction[:, :, i].T < threshold, extent=plot_extent, origin="lower", aspect="auto", cmap="OrRd"
            )
            plt.ylabel("Mean intensity (log10)")
            plt.xlabel("Minimum distance")
            if i == 0:
                plt.title("Accepted pixels")

        return fig


class AdaptiveFilterMisidentificationVsAbundance(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all"})

    def create_plot(self, **kwargs):
        filter_task = kwargs["tasks"]["filter_task"]
        adaptive_task = filter_task.get_adaptive_thresholds()

        fig = plt.figure(figsize=(7, 7))
        thresholds = np.arange(0.01, 0.5, 0.01)
        counts = [adaptive_task.calculate_barcode_count_for_threshold(x) for x in thresholds]
        misidentification_rates = [adaptive_task.calculate_misidentification_rate_for_threshold(x) for x in thresholds]
        plt.plot(misidentification_rates, counts, ".")

        selected_misid = filter_task.parameters["misidentification_rate"]
        selected_threshold = adaptive_task.calculate_threshold_for_misidentification_rate(selected_misid)
        selected_count = adaptive_task.calculate_barcode_count_for_threshold(selected_threshold)
        plt.scatter([selected_misid], [selected_count], s=20, facecolors="none", edgecolors="r")
        plt.ylabel("Barcode count")
        plt.xlabel("Misidentification rate")
        plt.title("Abundance vs misidentification rate")

        return fig


class AdaptiveFilterCountsPerArea(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"filter_task": "all"})

    def create_plot(self, **kwargs):
        filter_task = kwargs["tasks"]["filter_task"]
        adaptive_task = filter_task.get_adaptive_thresholds()

        threshold = adaptive_task.calculate_threshold_for_misidentification_rate(
            filter_task.parameters["misidentification_rate"]
        )
        blank_histogram = adaptive_task.get_blank_count_histogram()
        coding_histogram = adaptive_task.get_coding_count_histogram()
        blank_fraction = adaptive_task.get_blank_fraction_histogram()
        area_bins = adaptive_task.get_area_bins()
        all_counts_per_area = np.sum(blank_histogram + coding_histogram, axis=(0, 1))
        blank_histogram[blank_fraction >= threshold] = 0
        coding_histogram[coding_histogram >= threshold] = 0
        counts_per_area = np.sum(blank_histogram + coding_histogram, axis=(0, 1))

        fig = plt.figure(figsize=(15, 7))
        plt.bar(area_bins[:-1], np.log10(all_counts_per_area), width=1)
        plt.bar(area_bins[:-1], np.log10(counts_per_area), width=1)
        plt.legend(["All barcodes", "Filtered barcodes"])
        plt.ylabel("Barcode count (log10)")
        plt.xlabel("Area")
        plt.title("Abundance vs area")

        return fig


class FOVSpatialDistributionMetadata(PlotMetadata):
    def __init__(self, plot_task, required_tasks):
        super().__init__(plot_task, required_tasks)
        self.filterTask = self.required_tasks["filter_task"]

        dataset = self.plot_task.dataSet
        self.width = dataset.get_image_dimensions()[0]
        self.height = dataset.get_image_dimensions()[1]
        image_size = max(self.height, self.width)
        self.radial_bins = np.arange(0, 0.5 * image_size, (0.5 * image_size) / 200)
        self.spatial_x_bins = np.arange(0, self.width, 0.01 * self.width)
        self.spatial_y_bins = np.arange(0, self.height, 0.01 * self.height)
        self.multi_color_counts = np.zeros(len(self.radial_bins) - 1)
        self.single_color_counts = np.zeros(len(self.radial_bins) - 1)
        self.spatial_coding_counts = np.zeros((len(self.spatial_x_bins) - 1, len(self.spatial_y_bins) - 1))
        self.spatial_blank_counts = np.zeros((len(self.spatial_x_bins) - 1, len(self.spatial_y_bins) - 1))

        bit_colors = dataset.get_data_organization().data["color"]
        barcodes = self.filterTask.get_codebook().get_barcodes()
        self.single_color_barcodes = [i for i, b in enumerate(barcodes) if bit_colors[np.where(b)[0]].nunique() == 1]
        self.multi_color_barcodes = [i for i, b in enumerate(barcodes) if bit_colors[np.where(b)[0]].nunique() > 1]

        self.register_updaters({"filter_task": self.process_barcodes})
        self.register_datasets(
            "multi_color_counts", "single_color_counts", "spatial_coding_counts", "spatial_blank_counts"
        )

    def get_spatial_extents(self) -> List[float]:
        return [self.spatial_x_bins[0], self.spatial_x_bins[-1], self.spatial_y_bins[0], self.spatial_y_bins[-1]]

    def radial_distance(self, x: float, y: float) -> float:
        return np.sqrt((x - 0.5 * self.width) ** 2 + (y - 0.5 * self.height) ** 2)

    def radial_distribution(self, barcodes, barcode_ids):
        selected = barcodes[barcodes["barcode_id"].isin(barcode_ids)]
        distances = [self.radial_distance(r["x"], r["y"]) for _, r in selected.iterrows()]
        return np.histogram(distances, bins=self.radial_bins)[0]

    def spatial_distribution(self, barcodes, barcode_ids):
        selected = barcodes[barcodes["barcode_id"].isin(barcode_ids)]
        if len(selected) > 1:
            return np.histogram2d(selected["x"], selected["y"], bins=(self.spatial_x_bins, self.spatial_y_bins))[0]
        return 0

    def process_barcodes(self, fov) -> None:
        codebook = self.filterTask.get_codebook()
        barcodes = self.filterTask.load_result("barcodes", fov)
        barcodes = pd.DataFrame(barcodes[:, [-1, 5, 6]], columns=["barcode_id", "x", "y"])
        if len(barcodes) > 0:
            self.spatial_coding_counts += self.spatial_distribution(barcodes, codebook.get_coding_indexes())
            self.spatial_blank_counts += self.spatial_distribution(barcodes, codebook.get_blank_indexes())
            self.single_color_counts += self.radial_distribution(barcodes, self.single_color_barcodes)
            self.multi_color_counts += self.radial_distribution(barcodes, self.multi_color_barcodes)


class FilteredBarcodesMetadata(PlotMetadata):
    def __init__(self, plot_task, required_tasks):
        super().__init__(plot_task, required_tasks)
        self.filter_task = self.required_tasks["filter_task"]
        codebook = self.filter_task.get_codebook()

        self.barcode_counts = np.zeros(codebook.get_barcode_count())

        self.register_updaters({"filter_task": self.process_barcodes})
        self.register_datasets("barcode_counts")

    def process_barcodes(self, fov) -> None:
        barcodes = self.filter_task.load_result("barcodes", fov)[:, -1]
        self.barcode_counts += np.histogram(barcodes, bins=np.arange(len(self.barcode_counts) + 1))[0]


class GlobalSpatialDistributionMetadata(PlotMetadata):
    def __init__(self, plot_task, required_tasks):
        super().__init__(plot_task, required_tasks)
        # filter_task = self.required_tasks["filter_task"]
        global_task = self.required_tasks["global_align_task"]
        min_x, min_y, max_x, max_y = global_task.get_global_extent()
        x_step = (max_x - min_x) / 1000
        y_step = (max_x - min_x) / 1000
        # codebook = filter_task.get_codebook()

        # self.barcode_counts = self._load_numpy_metadata("barcode_counts", np.zeros(codebook.get_barcode_count()))
        self.spatial_x_bins = np.arange(min_x, max_x, x_step)
        self.spatial_y_bins = np.arange(min_y, max_y, y_step)
        self.spatial_coding_counts = np.zeros((len(self.spatial_x_bins) - 1, len(self.spatial_y_bins) - 1))
        self.spatial_blank_counts = np.zeros((len(self.spatial_x_bins) - 1, len(self.spatial_y_bins) - 1))

        self.register_updaters({"filter_task": self.process_barcodes})
        self.register_datasets("spatial_coding_counts", "spatial_blank_counts")

    def spatial_distribution(self, barcodes, barcode_ids):
        selected = barcodes[barcodes["barcode_id"].isin(barcode_ids)]
        if len(selected) > 1:
            return np.histogram2d(
                selected["global_x"], selected["global_y"], bins=(self.spatial_x_bins, self.spatial_y_bins)
            )[0]
        return 0

    def get_spatial_extents(self) -> List[float]:
        global_task = self.required_tasks["global_align_task"]
        min_x, min_y, max_x, max_y = global_task.get_global_extent()
        return [min_x, max_x, min_y, max_y]

    def process_barcodes(self, fov) -> None:
        filter_task = self.required_tasks["filter_task"]
        codebook = filter_task.get_codebook()
        barcodes = filter_task.load_result("barcodes", fov)
        barcodes = pd.DataFrame(barcodes[:, [-1, 8, 9]], columns=["barcode_id", "global_x", "global_y"])
        self.spatial_coding_counts += self.spatial_distribution(barcodes, codebook.get_coding_indexes())
        self.spatial_blank_counts += self.spatial_distribution(barcodes, codebook.get_blank_indexes())
