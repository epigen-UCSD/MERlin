from pathlib import Path
import time

import numpy as np
import pandas
from scipy import optimize

from merlin.core import analysistask
from merlin.analysis import decode


class AbstractFilterBarcodes(decode.BarcodeSavingParallelAnalysisTask):
    """
    An abstract class for filtering barcodes identified by pixel-based decoding.
    """

    def setup(self, *, parallel: bool) -> None:
        super().setup(parallel=parallel)

    def get_codebook(self):
        return self.decode_task.get_codebook()


class FilterBarcodes(AbstractFilterBarcodes):
    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"decode_task": []})
        self.set_default_parameters({"area_threshold": 3, "intensity_threshold": 200, "distance_threshold": 1e6})
        self.define_results("barcodes")

    def run_analysis(self):
        areaThreshold = self.parameters["area_threshold"]
        intensityThreshold = self.parameters["intensity_threshold"]
        distanceThreshold = self.parameters["distance_threshold"]
        self.barcodes = self.decode_task.load_result("barcodes")
        self.barcodes = self.barcodes[self.barcodes[:, 0] >= intensityThreshold]
        self.barcodes = self.barcodes[self.barcodes[:, 2] >= areaThreshold]
        self.barcodes = self.barcodes[self.barcodes[:, 4] <= intensityThreshold]


class GenerateAdaptiveThreshold(analysistask.AnalysisTask):
    """
    An analysis task that generates a three-dimension mean intenisty,
    area, minimum distance histogram for barcodes as they are decoded.
    """

    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies({"run_after_task": []})
        self.set_default_parameters({"tolerance": 0.001})

        self.define_results("blank_counts", "coding_counts", "area_bins", "distance_bins", "intensity_bins")

        self.decode_task = self.dataSet.load_analysis_task(self.parameters["decode_task"], fragment="")

    def get_blank_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result("blank_counts", self)

    def get_coding_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result("coding_counts", self)

    def get_total_count_histogram(self) -> np.ndarray:
        return self.get_blank_count_histogram() + self.get_coding_count_histogram()

    def get_area_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result("area_bins", self)

    def get_distance_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result("distance_bins", self)

    def get_intensity_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result("intensity_bins", self, None)

    def get_blank_fraction_histogram(self) -> np.ndarray:
        """Get the normalized blank fraction histogram indicating the
        normalized blank fraction for each intensity, distance, and area
        bin.

        Returns: The normalized blank fraction histogram. The histogram
            has three dimensions: mean intensity, minimum distance, and area.
            The bins in each dimension are defined by the bins returned by
            get_area_bins, get_distance_bins, and get_area_bins, respectively.
            Each entry indicates the number of blank barcodes divided by the
            number of coding barcodes within the corresponding bin
            normalized by the fraction of blank barcodes in the codebook.
            With this normalization, when all (both blank and coding) barcodes
            are selected with equal probability, the blank fraction is
            expected to be 1.
        """
        blankHistogram = self.get_blank_count_histogram()
        totalHistogram = self.get_coding_count_histogram()
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            blankFraction = blankHistogram / totalHistogram
            codebook = self.decode_task.get_codebook()
            blankBarcodeCount = len(codebook.get_blank_indexes())
            codingBarcodeCount = len(codebook.get_coding_indexes())
            blankFraction /= blankBarcodeCount / (blankBarcodeCount + codingBarcodeCount)
        blankFraction[totalHistogram == 0] = np.finfo(blankFraction.dtype).max
        return blankFraction

    def calculate_misidentification_rate_for_threshold(self, threshold: float) -> float:
        """Calculate the misidentification rate for a specified blank
        fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The estimated misidentification rate, estimated as the
            number of blank barcodes per blank barcode divided
            by the number of coding barcodes per coding barcode.
        """
        codebook = self.decode_task.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()

        selectBins = blankFraction < threshold
        codingCounts = np.sum(codingHistogram[selectBins])
        blankCounts = np.sum(blankHistogram[selectBins])

        return (blankCounts / blankBarcodeCount) / (codingCounts / codingBarcodeCount)

    def calculate_threshold_for_misidentification_rate(self, targetMisidentificationRate: float) -> float:
        """Calculate the normalized blank fraction threshold that achieves
        a specified misidentification rate.

        Args:
            targetMisidentificationRate: the target misidentification rate
        Returns: the normalized blank fraction threshold that achieves
            targetMisidentificationRate
        """
        tolerance = self.parameters["tolerance"]

        def misidentification_rate_error_for_threshold(x, targetError):
            return self.calculate_misidentification_rate_for_threshold(x) - targetError

        return optimize.newton(
            misidentification_rate_error_for_threshold,
            0.2,
            args=[targetMisidentificationRate],
            tol=tolerance,
            x1=0.3,
            disp=False,
        )

    def calculate_barcode_count_for_threshold(self, threshold: float) -> float:
        """Calculate the number of barcodes remaining after applying
        the specified normalized blank fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The number of barcodes passing the threshold.
        """
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()
        return np.sum(blankHistogram[blankFraction < threshold]) + np.sum(codingHistogram[blankFraction < threshold])

    def extract_barcodes_with_threshold(self, blankThreshold: float, barcodeSet: pandas.DataFrame) -> pandas.DataFrame:
        selectData = barcodeSet[:, [0, 4, 2]]
        selectData[:, 0] = np.log10(selectData[:, 0])
        blankFractionHistogram = self.get_blank_fraction_histogram()

        barcodeBins = (
            np.array(
                (
                    np.digitize(selectData[:, 0], self.get_intensity_bins(), right=True),
                    np.digitize(selectData[:, 1], self.get_distance_bins(), right=True),
                    np.digitize(selectData[:, 2], self.get_area_bins()),
                )
            )
            - 1
        )
        barcodeBins[0, :] = np.clip(barcodeBins[0, :], 0, blankFractionHistogram.shape[0] - 1)
        barcodeBins[1, :] = np.clip(barcodeBins[1, :], 0, blankFractionHistogram.shape[1] - 1)
        barcodeBins[2, :] = np.clip(barcodeBins[2, :], 0, blankFractionHistogram.shape[2] - 1)
        raveledIndexes = np.ravel_multi_index(barcodeBins[:, :], blankFractionHistogram.shape)

        thresholdedBlankFraction = blankFractionHistogram < blankThreshold
        return barcodeSet[np.take(thresholdedBlankFraction, raveledIndexes)]

    @staticmethod
    def _extract_counts(barcodes, intensityBins, distanceBins, areaBins):
        barcodeData = barcodes[:, [0, 4, 2]]
        barcodeData[:, 0] = np.log10(barcodeData[:, 0])
        return np.histogramdd(barcodeData, bins=(intensityBins, distanceBins, areaBins))[0]

    def run_analysis(self):
        codebook = self.decode_task.get_codebook()

        self.complete_fragments = self.dataSet.load_numpy_analysis_result_if_available(
            "complete_fragments", self, [False] * len(self.dataSet.get_fovs())
        )
        pending_fragments = []
        for i, fragment in enumerate(self.dataSet.get_fovs()):
            self.decode_task.fragment = fragment
            if self.decode_task.is_complete() and not self.complete_fragments[i]:
                pending_fragments.append(True)
            else:
                pending_fragments.append(False)

        self.area_bins = self.dataSet.load_numpy_analysis_result_if_available("area_bins", self, np.arange(1, 35))
        self.distance_bins = self.dataSet.load_numpy_analysis_result_if_available(
            "distance_bins", self, np.arange(0, self.decode_task.parameters["distance_threshold"] + 0.02, 0.01)
        )
        self.intensity_bins = self.dataSet.load_numpy_analysis_result_if_available("intensity_bins", self, None)

        self.blank_counts = self.dataSet.load_numpy_analysis_result_if_available("blank_counts", self, None)
        self.coding_counts = self.dataSet.load_numpy_analysis_result_if_available("coding_counts", self, None)

        self.save_result("area_bins")
        self.save_result("distance_bins")

        updated = False
        while not all(self.complete_fragments):
            time.sleep(10)
            if self.intensity_bins is None or self.blank_counts is None or self.coding_counts is None:
                for i, fragment in enumerate(self.dataSet.get_fovs()):
                    self.decode_task.fragment = fragment
                    if not pending_fragments[i] and self.decode_task.is_complete():
                        pending_fragments[i] = self.decode_task.is_complete()

                if np.sum(pending_fragments) >= min(20, len(self.dataSet.get_fovs())):

                    def extreme_values(inputData: pandas.Series):
                        return inputData.min(), inputData.max()

                    sampledFragments = np.random.choice(
                        [self.dataSet.get_fovs()[i] for i, p in enumerate(pending_fragments) if p], size=20
                    )
                    intensityExtremes = [
                        extreme_values(self.decode_task.load_result("barcodes", fragment)[:, 0])
                        for fragment in sampledFragments
                    ]
                    maxIntensity = np.log10(np.max([x[1] for x in intensityExtremes]))
                    self.intensity_bins = np.arange(0, 2 * maxIntensity, maxIntensity / 100)
                    self.save_result("intensity_bins")

                    self.blank_counts = np.zeros(
                        (len(self.intensity_bins) - 1, len(self.distance_bins) - 1, len(self.area_bins) - 1)
                    )
                    self.coding_counts = np.zeros(
                        (len(self.intensity_bins) - 1, len(self.distance_bins) - 1, len(self.area_bins) - 1)
                    )

            else:
                for i, fragment in enumerate(self.dataSet.get_fovs()):
                    self.decode_task.fragment = fragment
                    if not self.complete_fragments[i] and self.decode_task.is_complete():
                        barcodes = self.decode_task.load_result("barcodes", fragment)
                        self.blank_counts += self._extract_counts(
                            barcodes[np.isin(barcodes[:, -1], codebook.get_blank_indexes())],
                            self.intensity_bins,
                            self.distance_bins,
                            self.area_bins,
                        )
                        self.coding_counts += self._extract_counts(
                            barcodes[np.isin(barcodes[:, -1], codebook.get_coding_indexes())],
                            self.intensity_bins,
                            self.distance_bins,
                            self.area_bins,
                        )
                        updated = True
                        self.complete_fragments[i] = True

                if updated:
                    self.save_result("complete_fragments")
                    self.save_result("blank_counts")
                    self.save_result("coding_counts")


class AdaptiveFilterBarcodes(AbstractFilterBarcodes):
    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"adaptive_task": [], "decode_task": []})
        self.set_default_parameters({"misidentification_rate": 0.05})
        self.define_results("barcodes")

    def get_adaptive_thresholds(self):
        """Get the adaptive thresholds used for filtering barcodes.

        Returns: The GenerateaAdaptiveThershold task using for this
            adaptive filter.
        """
        return self.adaptive_task

    def run_analysis(self):
        threshold = self.adaptive_task.calculate_threshold_for_misidentification_rate(
            self.parameters["misidentification_rate"]
        )

        currentBarcodes = self.decode_task.load_result("barcodes", self.fragment)

        self.barcodes = self.adaptive_task.extract_barcodes_with_threshold(threshold, currentBarcodes)

    def metadata(self) -> dict:
        pixels = np.prod(self.dataSet.get_image_dimensions()) * len(self.dataSet.get_z_positions())
        blanks = self.barcodes[np.isin(self.barcodes[:, -1], self.get_codebook().get_blank_indexes())]
        genes = self.barcodes[np.isin(self.barcodes[:, -1], self.get_codebook().get_coding_indexes())]
        unfiltered = self.decode_task.load_result("barcodes", self.fragment)
        unfiltered_genes = unfiltered[np.isin(unfiltered[:, -1], self.get_codebook().get_coding_indexes())]
        try:
            filtered_fraction = 1 - (len(self.barcodes) / len(unfiltered))
            filtered_gene_fraction = 1 - (len(genes) / len(unfiltered_genes))
            filtered_pixel_fraction = 1 - (genes[:, 2].sum() / unfiltered_genes[:, 2].sum())
            gene_pixel_fraction = genes[:, 2].sum() / pixels
        except ZeroDivisionError:
            filtered_fraction = 0
            filtered_gene_fraction = 0
            filtered_pixel_fraction = 0
        return {
            "total": len(self.barcodes),
            "blanks": len(blanks),
            "genes": len(genes),
            "gene_pixel_fraction": gene_pixel_fraction,
            "filtered_fraction": filtered_fraction,
            "filtered_gene_fraction": filtered_gene_fraction,
            "filtered_pixel_fraction": filtered_pixel_fraction,
        }
