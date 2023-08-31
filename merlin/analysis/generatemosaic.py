from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from merlin.core import analysistask

ExtentTuple = Tuple[float, float, float, float]


class GenerateMosaic(analysistask.AnalysisTask):
    """An analysis task that generates mosaic images by compiling different field of views."""

    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies({"global_align_task": []})
        self.set_default_parameters({"microns_per_pixel": 3, "channel": "PolyT", "z_index": "max_projection"})

        if self.parameters["microns_per_pixel"] == "full_resolution":
            self.mosaic_microns_per_pixel = self.dataSet.get_microns_per_pixel()
        else:
            self.mosaic_microns_per_pixel = self.parameters["microns_per_pixel"]

    def micron_to_mosaic_pixel(self, micronCoordinates, micronExtents) -> np.ndarray:
        """Calculate the mosaic coordinates in pixels from the specified global coordinates."""
        return np.matmul(self.micron_to_mosaic_transform(micronExtents), np.append(micronCoordinates, 1)).astype(
            np.int32
        )[:2]

    def micron_to_mosaic_transform(self, micronExtents: ExtentTuple) -> np.ndarray:
        s = 1 / self.mosaic_microns_per_pixel
        return np.float32([[s * 1, 0, -s * micronExtents[0]], [0, s * 1, -s * micronExtents[1]], [0, 0, 1]])

    def run_analysis(self):
        micron_extents = self.global_align_task.get_global_extent()
        self.dataSet.save_numpy_txt_analysis_result(
            self.micron_to_mosaic_transform(micron_extents), "micron_to_mosaic_pixel_transform", self
        )

        data_organization = self.dataSet.get_data_organization()

        if isinstance(self.parameters["channel"], list):
            channel_names = self.parameters["channel"]
            channels = [
                data_organization.get_data_channel_index(channel) if isinstance(channel, str) else channel
                for channel in self.parameters["channel"]
            ]
        elif isinstance(self.parameters["channel"], str):
            if self.parameters["channel"] == "all":
                channels = data_organization.get_data_channels()
                channel_names = [data_organization.get_data_channel_name(c) for c in channels]
            else:
                channel_names = [self.parameters["channel"]]
                channels = [data_organization.get_data_channel_index(self.parameters["channel"])]
        else:
            channel_names = [self.parameters["channel"]]
            channels = [self.parameters["channel"]]

        for channel, name in zip(channels, channel_names):
            mosaic = self.create_mosaic(channel, micron_extents)
            np.save(self.result_path(f"mosaic_{name}", ".npy"), mosaic)
            plt.imsave(
                self.result_path(f"mosaic_{name}", ".png"), mosaic, cmap="gray", vmax=np.percentile(mosaic, 99.9)
            )

    def create_mosaic(self, channel, micron_extents):
        mosaic_dimensions = tuple(self.micron_to_mosaic_pixel(micron_extents[-2:], micron_extents))

        mosaic = np.zeros(np.flip(mosaic_dimensions, axis=0), dtype=np.uint16)

        for fov in self.dataSet.get_fovs():
            if self.parameters["z_index"] == "max_projection":
                image = np.max(
                    [
                        self.dataSet.get_raw_image(channel, fov, self.dataSet.z_index_to_position(z))
                        for z in range(len(self.dataSet.get_z_positions()))
                    ],
                    axis=0,
                )
            else:
                image = self.dataSet.get_raw_image(
                    channel, fov, self.dataSet.z_index_to_position(self.parameters["z_index"])
                )

            overlap_mask = self.dataSet.get_overlap_mask(fov, trim=True)
            image[overlap_mask == 1] = 0

            sizex = (
                self.dataSet.get_image_dimensions()[0] * self.dataSet.get_microns_per_pixel()
            ) / self.mosaic_microns_per_pixel
            sizey = (
                self.dataSet.get_image_dimensions()[1] * self.dataSet.get_microns_per_pixel()
            ) / self.mosaic_microns_per_pixel

            image = cv2.resize(image, (int(sizex), int(sizey)))

            x, y = self.micron_to_mosaic_pixel(
                self.global_align_task.fov_coordinates_to_global(fov, (0, 0)), micron_extents
            )
            x2 = x + image.shape[0]
            y2 = y + image.shape[1]

            mosaic[x:x2, y:y2][image > 0] = image[image > 0]

        return mosaic
