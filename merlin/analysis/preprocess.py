"""Analysis tasks for performing image pre-processing."""

from functools import cached_property

import cv2
import numpy as np

from merlin.core import analysistask
from merlin.data import codebook
from merlin.util import aberration, deconvolve, imagefilters


class DeconvolutionPreprocess(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"warp_task": ["drifts"]})
        self.add_dependencies({"flat_field_task": []}, optional=True)

        self.set_default_parameters(
            {"highpass_sigma": 3, "decon_sigma": 2, "decon_iterations": 20, "codebook_index": 0, "lowpass_sigma": 0}
        )

        if "decon_filter_size" not in self.parameters:
            self.parameters["decon_filter_size"] = int(2 * np.ceil(2 * self.parameters["decon_sigma"]) + 1)

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters["codebook_index"])

    def get_processed_image_set(
        self, z_index: int = None, chromatic_corrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if z_index is None:
            return np.array(
                [
                    [
                        self.get_processed_image(
                            self.dataSet.get_data_organization().get_data_channel_for_bit(b),
                            zIndex,
                            chromatic_corrector,
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
                        self.dataSet.get_data_organization().get_data_channel_for_bit(b),
                        z_index,
                        chromatic_corrector,
                    )
                    for b in self.get_codebook().get_bit_names()
                ]
            )

    def get_processed_image(
        self, data_channel: int, z_index: int, chromatic_corrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        input_image = self.warp_task.get_z_aligned_frame(data_channel, z_index)
        if "flat_field_task" in self.dependencies:
            input_image = self.flat_field_task.process_image(
                input_image, self.dataSet.get_data_organization().get_data_channel_color(data_channel)
            )
        processed_image = self.process_image(input_image)
        return self.warp_task.align_image(data_channel, processed_image, chromatic_corrector)

    def high_pass_filter(self, image: np.ndarray) -> np.ndarray:
        highpass_sigma = self.parameters["highpass_sigma"]
        filter_size = int(2 * np.ceil(2 * highpass_sigma) + 1)
        hp_image = imagefilters.high_pass_filter(image, filter_size, highpass_sigma)
        return hp_image.astype(np.float32)

    def process_image(self, image: np.ndarray) -> np.ndarray:
        filter_size = self.parameters["decon_filter_size"]

        if self.parameters["lowpass_sigma"] > 0:
            image = cv2.GaussianBlur(image, (21, 21), self.parameters["lowpass_sigma"])
        filtered_image = self.high_pass_filter(image)
        if self._deconIterations > 0:
            return deconvolve.deconvolve_lucyrichardson(
                filtered_image, filter_size, self.parameters["decon_sigma"], self.parameters["decon_iterations"]
            ).astype(np.uint16)
        return filtered_image


class DeconvolutionPreprocessGuo(DeconvolutionPreprocess):
    def setup(self) -> None:
        super().setup()

        # Check for 'decon_iterations' in parameters instead of
        # self.parameters as 'decon_iterations' is added to
        # self.parameters by the super-class with a default value
        # of 20, but we want the default value to be 2.
        # if "decon_iterations" not in parameters:
        #    self.parameters["decon_iterations"] = 2

    def process_image(self, image: np.ndarray) -> np.ndarray:
        filter_size = self.parameters["decon_filter_size"]
        filtered_image = self.high_pass_filter(image)
        return deconvolve.deconvolve_lucyrichardson_guo(
            filtered_image, filter_size, self.parameters["decon_sigma"], self.parameters["decon_iterations"]
        ).astype(np.uint16)


class DeconvolutionSdeconv(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True, threads=8)

        self.add_dependencies({"warp_task": ["drifts"]})
        self.set_default_parameters({"highpass_sigma": 3, "codebook_index": 0})

        self._highPassSigma = self.parameters["highpass_sigma"]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters["codebook_index"])

    def get_processed_image_set(
        self, zIndex: int = None, chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        return np.array(
            [
                self.get_processed_image(
                    self.dataSet.get_data_organization().get_data_channel_for_bit(b),
                    zIndex,
                    chromaticCorrector,
                )
                for b in self.get_codebook().get_bit_names()
            ]
        )

    def get_processed_image(
        self, dataChannel: int, zIndex: int = None, chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warp_task.get_aligned_image(dataChannel, zIndex, chromaticCorrector)
        return self._preprocess_image(inputImage)

    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        if inputImage.ndim >= 3:
            return np.array([self._high_pass_filter(img) for img in inputImage])
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage, highPassFilterSize, self._highPassSigma)
        return hpImage.astype(np.float32)

    @cached_property
    def psf(self):
        return np.load(self.parameters["psf_file"])

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        inputImage = deconvolve.deconvolve_sdeconv(inputImage, self.psf)
        return self._high_pass_filter(inputImage)


class FlatFieldPreprocess(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.define_results("mean_image")

        self.fragment_list = self.dataSet.dataOrganization.get_data_colors(merfish_only=False)

    @cached_property
    def mean_image(self) -> np.ndarray:
        return self.load_result("mean_image")

    def process_image(self, image: np.ndarray, color: str = None, channel: int = None) -> np.ndarray:
        if color is None:
            color = self.dataSet.get_data_organization().get_data_channel_color(channel)
        self.fragment = str(color)
        return (image / self.mean_image) * np.median(self.mean_image)

    def run_analysis(self) -> None:
        for channel in self.dataSet.get_data_organization().get_data_channels():
            if self.dataSet.get_data_organization().get_data_channel_color(channel) == self.fragment:
                break
        sum_image = np.zeros(self.dataSet.get_image_dimensions(), dtype=np.uint32)
        zlist = self.dataSet.get_z_positions()
        middle_z = zlist[len(zlist) // 2]
        for fov in self.dataSet.get_fovs():
            sum_image += self.dataSet.get_raw_image(channel, fov, middle_z)
        self.mean_image = sum_image / len(self.dataSet.get_fovs())
