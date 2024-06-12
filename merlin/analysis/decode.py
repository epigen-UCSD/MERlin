import numpy as np

from merlin.core import analysistask
from merlin.data.codebook import Codebook
from merlin.util import barcodedb, barcodefilters, decoding


class BarcodeSavingParallelAnalysisTask(analysistask.AnalysisTask):
    """An abstract analysis class that saves barcodes into a barcode database."""

    def setup(self, *, parallel: bool, threads: int = 1) -> None:
        super().setup(parallel=parallel, threads=threads)

    def reset_analysis(self) -> None:
        super().reset_analysis()
        self.get_barcode_database().empty_database(self.fragment)

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        """Get the barcode database this analysis task saves barcodes into.

        Returns: The barcode database reference.
        """
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)


class Decode(BarcodeSavingParallelAnalysisTask):
    """An analysis task that extracts barcodes from images."""

    def setup(self) -> None:
        super().setup(parallel=True, threads=16)

        self.add_dependencies(
            {
                "preprocess_task": [],
                "optimize_task": ["scale_factors", "background_factors", "chromatic_corrections"],
                "global_align_task": [],
            }
        )

        self.define_results("barcodes")

        self.set_default_parameters(
            {
                "crop_width": 100,
                "minimum_area": 0,
                "distance_threshold": 0.5167,
                "lowpass_sigma": 1,
                "decode_3d": False,
                "memory_map": False,
                "remove_z_duplicated_barcodes": False,
                "z_duplicate_zPlane_threshold": 1,
                "z_duplicate_xy_pixel_threshold": np.sqrt(2),
            }
        )

        self.crop_width = self.parameters["crop_width"]
        self.image_size = self.dataSet.get_image_dimensions()

    def get_codebook(self) -> Codebook:
        return self.preprocess_task.get_codebook()

    def run_analysis(self):
        """This function decodes the barcodes in a fov and saves them to the barcode database."""
        decoder = decoding.PixelBasedDecoder(self.get_codebook())
        scale_factors = self.optimize_task.load_result("scale_factors")
        backgrounds = self.optimize_task.load_result("background_factors")
        chromatic_corrector = self.optimize_task.get_chromatic_corrector()

        shape = (len(self.dataSet.get_z_positions()), *self.dataSet.get_image_dimensions())
        decoded_images = np.zeros(shape, dtype=np.int16)
        magnitude_images = np.zeros(shape, dtype=np.float32)
        distances = np.zeros(shape, dtype=np.float32)
        self.barcodes = np.array([], dtype=np.float32).reshape((0, 12 + self.get_codebook().get_bit_count()))

        if not self.parameters["decode_3d"]:
            for z_index in range(len(self.dataSet.get_z_positions())):
                di, pm, d = self._process_independent_z_slice(
                    z_index,
                    chromatic_corrector,
                    scale_factors,
                    backgrounds,
                    self.preprocess_task,
                    decoder,
                )

                decoded_images[z_index, :, :] = di
                magnitude_images[z_index, :, :] = pm
                distances[z_index, :, :] = d

        else:
            for z_index in range(len(self.dataSet.get_z_positions())):
                image_set = self.preprocess_task.get_processed_image_set(z_index, chromatic_corrector)
                image_set = image_set.reshape((image_set.shape[0], image_set.shape[-2], image_set.shape[-1]))

                di, pm, _, d = decoder.decode_pixels(
                    image_set,
                    scale_factors,
                    backgrounds,
                    lowPassSigma=self.parameters["lowpass_sigma"],
                    distanceThreshold=self.parameters["distance_threshold"],
                )

                decoded_images[z_index, :, :] = di
                magnitude_images[z_index, :, :] = pm
                distances[z_index, :, :] = d

            self.barcodes = decoder.extract_all_barcodes(
                decoded_images,
                magnitude_images,
                None,
                distances,
                self.fragment,
                self.crop_width,
                z_index,
                self.global_align_task,
                self.parameters["minimum_area"],
                quick_mode=True,
            )

        if self.parameters["remove_z_duplicated_barcodes"]:
            bcDB = self.get_barcode_database()
            bc = self._remove_z_duplicate_barcodes(bcDB.get_barcodes(fov=self.fragment))
            bcDB.empty_database(self.fragment)
            bcDB.write_barcodes(bc, fov=self.fragment)

    def metadata(self) -> dict:
        pixels = np.prod(self.dataSet.get_image_dimensions()) * len(self.dataSet.get_z_positions())
        blanks = self.barcodes[np.isin(self.barcodes[:, -1], self.get_codebook().get_blank_indexes())]
        genes = self.barcodes[np.isin(self.barcodes[:, -1], self.get_codebook().get_coding_indexes())]
        return {
            "total": len(self.barcodes),
            "blanks": len(blanks),
            "genes": len(genes),
            "total_pixel_fraction": self.barcodes[:, 2].sum() / pixels,
            "blank_pixel_fraction": blanks[:, 2].sum() / pixels,
            "gene_pixel_fraction": genes[:, 2].sum() / pixels,
            "blank_area": blanks[:, 2].mean(),
            "gene_area": genes[:, 2].mean(),
            "blank_mean_intensity": blanks[:, 0].mean(),
            "gene_mean_intensity": genes[:, 0].mean(),
            "blank_max_intensity": blanks[:, 1].mean(),
            "gene_max_intensity": genes[:, 1].mean(),
            "blank_min_distance": blanks[:, 4].mean(),
            "gene_min_distance": genes[:, 4].mean(),
            "blank_mean_distance": blanks[:, 3].mean(),
            "gene_mean_distance": genes[:, 3].mean(),
        }

    def _process_independent_z_slice(
        self, z_index: int, chromatic_corrector, scale_factors, backgrounds, preprocess_task, decoder
    ):
        image_set = preprocess_task.get_processed_image_set(z_index, chromatic_corrector)
        image_set = image_set.reshape((image_set.shape[0], image_set.shape[-2], image_set.shape[-1]))

        di, pm, npt, d = decoder.decode_pixels(
            image_set,
            scale_factors,
            backgrounds,
            lowPassSigma=self.parameters["lowpass_sigma"],
            distanceThreshold=self.parameters["distance_threshold"],
        )
        self.barcodes = np.vstack(
            [
                self.barcodes,
                decoder.extract_all_barcodes(
                    di,
                    pm,
                    npt,
                    d,
                    self.fragment,
                    self.crop_width,
                    z_index,
                    self.global_align_task,
                    self.parameters["minimum_area"],
                ),
            ]
        )

        return di, pm, d

    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc,
            self.parameters["z_duplicate_zPlane_threshold"],
            self.parameters["z_duplicate_xy_pixel_threshold"],
            self.dataSet.get_z_positions(),
        )
        return bc
