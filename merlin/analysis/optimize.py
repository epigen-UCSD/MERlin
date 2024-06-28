import itertools

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis import decode
from merlin.data.codebook import Codebook
from merlin.util import aberration, decoding, registration


def OptimizeTask(dataset, analysis_path, parameters, name, fragment):
    """A task that expands into a sequence of OptimizeIteration tasks."""
    tasks = []
    iteration_name = None
    if not name:
        name = "OptimizeTask"
    for i in range(1, parameters["iterations"] + 1):
        if iteration_name:
            parameters["previous_iteration"] = iteration_name
        iteration_name = name if i == parameters["iterations"] else f"{name}_{i}"
        tasks.append(OptimizeIteration(dataset, analysis_path, parameters, iteration_name, fragment))
    return tasks


class OptimizeIteration(decode.BarcodeSavingParallelAnalysisTask):
    """An analysis task for performing a single iteration of scale factor optimization."""

    def setup(self) -> None:
        super().setup(parallel=True, threads=4)

        self.add_dependencies({"preprocess_task": [], "warp_task": ["drifts"]})
        self.add_dependencies({"previous_iteration": []}, optional=True)

        self.set_default_parameters(
            {
                "fov_per_iteration": 50,
                "area_threshold": 5,
                "optimize_background": False,
                "optimize_chromatic_correction": False,
                "crop_width": 0,
                "optimize_3d": False,
            }
        )

        self.define_results(
            "select_frame", "scale_factors", "background_factors", "chromatic_corrections", "barcode_counts", "barcodes"
        )
        self.define_results("scale_factors", "background_factors", "chromatic_corrections", final=True)

        if "fov_index" in self.parameters:
            self.parameters["fov_per_iteration"] = len(self.parameters["fov_index"])
        else:
            path = self.path / "select_frame"
            files = path.glob("select_frame_*")
            self.parameters["fov_index"] = [file.stem.split("select_frame_")[-1] for file in files]
            if len(self.parameters["fov_index"]) < self.parameters["fov_per_iteration"]:
                zIndices = list(range(len(self.dataSet.get_z_positions())))
                zIndices = zIndices[5:-5]  # Avoid artifacts from z-drift or blurry images
                combinations = set(itertools.product(self.dataSet.get_fovs(), zIndices))
                combinations -= {tuple(zslice.split("__")) for zslice in self.parameters["fov_index"]}
                for zslice in np.random.choice(
                    [f"{fovIndex}__{zIndex}" for fovIndex, zIndex in combinations],
                    size=self.parameters["fov_per_iteration"] - len(self.parameters["fov_index"]),
                    replace=False,
                ):
                    self.parameters["fov_index"].append(zslice)
        self.fragment_list = self.parameters["fov_index"]

    def get_codebook(self) -> Codebook:
        return self.preprocess_task.get_codebook()

    def run_analysis(self):
        codebook = self.get_codebook()

        fov_index, z_index = self.fragment.split("__")
        z_index = int(z_index)
        if self.parameters["optimize_3d"]:
            z_index = None

        scale_factors = self._get_previous_scale_factors()
        backgrounds = self._get_previous_backgrounds()
        chromatic_transformation = self._get_previous_chromatic_transformations()

        self.select_frame = np.array([fov_index, z_index])
        self.save_result("select_frame")

        chromatic_corrector = aberration.RigidChromaticCorrector(chromatic_transformation, self.get_reference_color())
        self.chromatic_corrections = chromatic_transformation
        preprocess_task = self.dataSet.load_analysis_task(self.parameters["preprocess_task"], fov_index)
        warped_images = preprocess_task.get_processed_image_set(z_index, chromatic_corrector)

        decoder = decoding.PixelBasedDecoder(codebook)
        area_threshold = self.parameters["area_threshold"]
        decoder.refactorAreaThreshold = area_threshold
        di, pm, npt, d = decoder.decode_pixels(warped_images, scale_factors, backgrounds)

        self.scale_factors, self.background_factors, self.barcode_counts = decoder.extract_refactors(
            di, pm, npt, extractBackgrounds=self.parameters["optimize_background"]
        )

        # TODO this saves the barcodes under fragment instead of fov
        # the barcodedb should be made more general
        crop_width = self.parameters["crop_width"]
        self.barcodes = decoder.extract_all_barcodes(
            di, pm, npt, d, fov_index, crop_width, z_index, minimumArea=area_threshold
        )

    def _get_used_colors(self) -> list[str]:
        data_organization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted(
            {
                data_organization.get_data_channel_color(data_organization.get_data_channel_for_bit(x))
                for x in codebook.get_bit_names()
            }
        )

    def _get_previous_scale_factors(self) -> np.ndarray:
        if "previous_iteration" not in self.parameters:
            return np.ones(self.get_codebook().get_bit_count(), dtype=np.float32)
        return self.previous_iteration.load_result("scale_factors")

    def _get_previous_backgrounds(self) -> np.ndarray:
        if "previous_iteration" not in self.parameters:
            return np.zeros(self.get_codebook().get_bit_count(), dtype=np.float32)
        return self.previous_iteration.load_result("background_factors")

    def _get_previous_chromatic_transformations(self) -> dict[str, dict[str, transform.SimilarityTransform]]:
        if "previous_iteration" not in self.parameters:
            colors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform() for v in colors if v >= u} for u in colors}
        return self.previous_iteration.load_result("chromatic_corrections")

    # TODO the next two functions could be in a utility class. Make a
    #  chromatic aberration utility class

    def get_reference_color(self) -> str:
        return min(self._get_used_colors())

    def get_chromatic_corrector(self) -> aberration.ChromaticCorrector:
        """Get the chromatic corrector estimated from this optimization iteration.

        Returns:
            The chromatic corrector.
        """
        return aberration.RigidChromaticCorrector(self.load_result("chromatic_corrections"), self.get_reference_color())

    def _get_chromatic_transformations(self) -> dict[str, dict[str, transform.SimilarityTransform]]:
        """Get the estimated chromatic corrections from this optimization iteration.

        Returns:
            a dictionary of dictionary of transformations for transforming
            the farther red colors to the most blue color. The transformation
            for transforming the farther red color, e.g. '750', to the
            farther blue color, e.g. '560', is found at result['560']['750']
        """
        if not self.parameters["optimize_chromatic_correction"]:
            colors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform() for v in colors if v >= u} for u in colors}

        # TODO - this is messy. It can be broken into smaller subunits and
        # most parts could be included in a chromatic aberration class
        previous_transformations = self._get_previous_chromatic_transformations()
        previous_corrector = aberration.RigidChromaticCorrector(previous_transformations, self.get_reference_color())
        codebook = self.get_codebook()
        data_organization = self.dataSet.get_data_organization()

        barcodes = {fragment: self.load_result("barcodes", fragment) for fragment in self.fragment_list}

        colors = self._get_used_colors()
        colorPairDisplacements = {u: {v: [] for v in colors if v >= u} for u in colors}

        for fragment, fovBarcodes in barcodes.items():
            fov = fragment.split("__")[0]
            zIndexes = np.unique(fovBarcodes[:, 7])
            for z in zIndexes:
                currentBarcodes = fovBarcodes[fovBarcodes[:, 7] == z]
                # TODO this can be moved to the run function for the task
                # so not as much repeated work is done when it is called
                # from many different tasks in parallel
                warpedImages = np.array(
                    [
                        self.dataSet.load_analysis_task(self.parameters["warp_task"], fov).get_aligned_image(
                            fov, data_organization.get_data_channel_for_bit(b), int(z), previous_corrector
                        )
                        for b in codebook.get_bit_names()
                    ]
                )

                for cBC in currentBarcodes:
                    onBits = np.where(codebook.get_barcode(cBC[-1]))[0]

                    # TODO this can be done by crop width when decoding
                    if (
                        cBC[5] > 10
                        and cBC[6] > 10
                        and warpedImages.shape[1] - cBC[5] > 10
                        and warpedImages.shape[2] - cBC[6] > 10
                    ):
                        refinedPositions = np.array(
                            [registration.refine_position(warpedImages[i, :, :], cBC[5], cBC[6]) for i in onBits]
                        )
                        for p in itertools.combinations(enumerate(onBits), 2):
                            c1 = data_organization.get_data_channel_color(p[0][1])
                            c2 = data_organization.get_data_channel_color(p[1][1])

                            if c1 < c2:
                                colorPairDisplacements[c1][c2].append(
                                    [
                                        np.array([cBC[5], cBC[6]]),
                                        refinedPositions[p[1][0]] - refinedPositions[p[0][0]],
                                    ]
                                )
                            else:
                                colorPairDisplacements[c2][c1].append(
                                    [
                                        np.array([cBC[5], cBC[6]]),
                                        refinedPositions[p[0][0]] - refinedPositions[p[1][0]],
                                    ]
                                )

        tForms = {}
        for k, v in colorPairDisplacements.items():
            tForms[k] = {}
            for k2, v2 in v.items():
                tForm = transform.SimilarityTransform()
                goodIndexes = [i for i, x in enumerate(v2) if not any(np.isnan(x[1])) and not any(np.isinf(x[1]))]
                tForm.estimate(
                    np.array([v2[i][0] for i in goodIndexes]), np.array([v2[i][0] + v2[i][1] for i in goodIndexes])
                )
                tForms[k][k2] = tForm + previous_transformations[k][k2]

        return tForms

    def get_scale_factors(self) -> np.ndarray:
        """Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the
            scale factor corresponding to the i'th bit.
        """
        refactors = np.array(self.aggregate_result("scale_factors"))
        # Don't rescale bits that were never seen
        refactors[refactors == 0] = 1
        previous_factors = self._get_previous_scale_factors()
        return np.nanmedian(np.multiply(refactors, previous_factors), axis=0)

    def get_backgrounds(self) -> np.ndarray:
        refactors = np.array(self.aggregate_result("background_factors"))
        previous_backgrounds = self._get_previous_backgrounds()
        previous_factors = self._get_previous_scale_factors()
        return np.nanmedian(np.add(previous_backgrounds, np.multiply(refactors, previous_factors)), axis=0)

    def get_scale_factor_history(self) -> np.ndarray:
        """Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            scale factor corresponding to the i'th bit in the j'th
            iteration.
        """
        if "previous_iteration" not in self.parameters:
            return np.array([self.load_result("scale_factors")])
        previous_history = self.previous_iteration.get_scale_factor_history()
        return np.append(previous_history, [self.load_result("scale_factors")], axis=0)

    def get_barcode_count_history(self) -> np.ndarray:
        """Get the set of barcode counts for each iteration of the
        optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            barcode count corresponding to the i'th barcode in the j'th
            iteration.
        """
        counts_mean = np.mean(self.aggregate_result("barcode_counts"), axis=0)
        if "previous_iteration" not in self.parameters:
            return np.array([counts_mean])
        previous_history = self.previous_iteration.get_barcode_count_history()
        return np.append(previous_history, [counts_mean], axis=0)

    def finalize_analysis(self) -> None:
        self.scale_factors = self.get_scale_factors()
        self.background_factors = self.get_backgrounds()
        self.chromatic_corrections = self._get_chromatic_transformations()
