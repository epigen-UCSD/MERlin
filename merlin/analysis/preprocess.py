import os
import cv2
import numpy as np
from functools import cached_property
from pathlib import Path

from merlin.core import analysistask
from merlin.util import deconvolve
from merlin.util import aberration
from merlin.util import imagefilters
from merlin.data import codebook


class Preprocess(analysistask.AnalysisTask):

    """
    An abstract class for preparing data for barcode calling.
    """

    def _image_name(self, fov):
        destPath = self.dataSet.get_analysis_subdirectory(self.analysis_name, subdirectory="preprocessed_images")
        return os.sep.join([destPath, "fov_" + str(fov) + ".tif"])

    def get_pixel_histogram(self, fov=None):
        if fov is not None:
            return self.dataSet.load_numpy_analysis_result("pixel_histogram", self.analysis_name, fov, "histograms")

        pixelHistogram = np.zeros(self.get_pixel_histogram(self.dataSet.get_fovs()[0]).shape)
        for f in self.dataSet.get_fovs():
            pixelHistogram += self.get_pixel_histogram(f)

        return pixelHistogram

    def _save_pixel_histogram(self, histogram, fov):
        self.dataSet.save_numpy_analysis_result(histogram, "pixel_histogram", self.analysis_name, fov, "histograms")


class DeconvolutionPreprocess(Preprocess):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"warp_task": ["drifts"]})
        self.set_default_parameters(
            {"highpass_sigma": 3, "decon_sigma": 2, "decon_iterations": 20, "codebook_index": 0}
        )

        if "decon_filter_size" not in self.parameters:
            self.parameters["decon_filter_size"] = int(2 * np.ceil(2 * self.parameters["decon_sigma"]) + 1)

        self._highPassSigma = self.parameters["highpass_sigma"]
        self._deconSigma = self.parameters["decon_sigma"]
        self._deconIterations = self.parameters["decon_iterations"]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters["codebook_index"])

    def get_processed_image_set(
        self, fov, zIndex: int = None, chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if zIndex is None:
            return np.array(
                [
                    [
                        self.get_processed_image(
                            fov,
                            self.dataSet.get_data_organization().get_data_channel_for_bit(b),
                            zIndex,
                            chromaticCorrector,
                        )
                        for zIndex in range(len(self.dataSet.get_z_positions()))
                    ]
                    for b in self.get_codebook().get_bit_names()
                ]
            )
        else:
            return np.array(
                [
                    self.get_processed_image(
                        fov,
                        self.dataSet.get_data_organization().get_data_channel_for_bit(b),
                        zIndex,
                        chromaticCorrector,
                    )
                    for b in self.get_codebook().get_bit_names()
                ]
            )

    def get_processed_image(
        self, fov: int, dataChannel: int, zIndex: int, chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warp_task.get_aligned_image(fov, dataChannel, zIndex, chromaticCorrector)
        return self._preprocess_image(inputImage)

    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage, highPassFilterSize, self._highPassSigma)
        return hpImage.astype(np.float32)

    def run_analysis(self):
        histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
        pixelHistogram = np.zeros((self.get_codebook().get_bit_count(), len(histogramBins) - 1))

        # this currently only is to calculate the pixel histograms in order
        # to estimate the initial scale factors. This is likely unnecessary
        for bi, b in enumerate(self.get_codebook().get_bit_names()):
            dataChannel = self.dataSet.get_data_organization().get_data_channel_for_bit(b)
            for i in range(len(self.dataSet.get_z_positions())):
                inputImage = self.warp_task.get_aligned_image(self.fragment, dataChannel, i)
                deconvolvedImage = self._preprocess_image(inputImage)

                pixelHistogram[bi, :] += np.histogram(deconvolvedImage, bins=histogramBins)[0]

        self._save_pixel_histogram(pixelHistogram, self.fragment)

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters["decon_filter_size"]

        filteredImage = self._high_pass_filter(inputImage)
        deconvolvedImage = deconvolve.deconvolve_lucyrichardson(
            filteredImage, deconFilterSize, self._deconSigma, self._deconIterations
        ).astype(np.uint16)
        return deconvolvedImage


class DeconvolutionPreprocessGuo(DeconvolutionPreprocess):
    def setup(self) -> None:
        super().setup()

        # Check for 'decon_iterations' in parameters instead of
        # self.parameters as 'decon_iterations' is added to
        # self.parameters by the super-class with a default value
        # of 20, but we want the default value to be 2.
        # if "decon_iterations" not in parameters:
        #    self.parameters["decon_iterations"] = 2

        self._deconIterations = self.parameters["decon_iterations"]

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters["decon_filter_size"]

        filteredImage = self._high_pass_filter(inputImage)
        deconvolvedImage = deconvolve.deconvolve_lucyrichardson_guo(
            filteredImage, deconFilterSize, self._deconSigma, self._deconIterations
        ).astype(np.uint16)
        return deconvolvedImage


class FlatFieldPreprocess(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=False)

        self.define_results(f"mean_image_{self.parameters['channel']}")

    @cached_property
    def mean_image(self):
        return self.load_result(f"mean_image_{self.parameters['channel']}")

    def process_image(self, image):
        return image / self.mean_image

    def run_analysis(self) -> None:
        sum_image = np.zeros(self.dataSet.get_image_dimensions(), dtype=np.uint32)
        for fov in self.dataSet.get_fovs():
            sum_image += self.dataSet.get_raw_image(
                self.dataSet.get_data_organization().get_data_channel_index(self.parameters["channel"]), fov, 10
            )  # TODO: remove hard-coded zindex
        setattr(self, f"mean_image_{self.parameters['channel']}", sum_image)
