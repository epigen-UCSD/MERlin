import csv
import os
import pathlib
import re
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NBitBase, NDArray

import merlin


def _parse_list(string: str, dtype: DTypeLike = np.float64) -> NDArray[Any]:
    sep = "," if "," in string else " "
    return np.fromstring(string.strip("[] "), dtype=dtype, sep=sep)


def _parse_int_list(string: str) -> NDArray[Any]:
    return _parse_list(string, dtype=int)


def _parse_optional_list(string: str) -> NDArray[Any] | int:
    if string.strip().startswith("["):
        return _parse_int_list(string)
    return int(string)


class InputDataError(Exception):
    pass


class DataOrganization:
    """A class to specify the organization of raw images in the original image files."""

    def __init__(
        self,
        dataset,
        path: pathlib.Path | None = None,
        fov_list: str = "",
        skip: list[str] | None = None,
    ) -> None:
        """Create a new DataOrganization for the data in the specified data set.

        If path is not specified, a previously stored DataOrganization
        is loaded from the dataset if it exists. If path is specified,
        the DataOrganization at the specified path is loaded and
        stored in the dataset, overwriting any previously stored
        DataOrganization.

        Raises InputDataError if the set of raw data is incomplete or the
        format of the raw data deviates from expectations.
        """
        self.dataset = dataset

        converters = {"frame": _parse_int_list, "zPos": _parse_list, "fiducialFrame": _parse_optional_list}
        if path:
            if not path.exists():
                path = merlin.DATA_ORGANIZATION_HOME / path

            self.data = pd.read_csv(path, converters=converters)
            self.data["readoutName"] = self.data["readoutName"].str.strip()
            self.dataset.save_dataframe_to_csv(self.data, "dataorganization", index=False)

        else:
            self.data = self.dataset.load_dataframe_from_csv("dataorganization", converters=converters)

        string_columns = [
            "readoutName",
            "channelName",
            "imageType",
            "imageRegExp",
            "fiducialImageType",
            "fiducialRegExp",
        ]
        self.data[string_columns] = self.data[string_columns].astype("str")
        self.fov_list = None
        if fov_list:
            with pathlib.Path(fov_list).open() as f:
                self.fov_list = [fov.strip() for fov in f.readlines()]
        self.skip = skip
        self._map_image_files()

    def get_data_channels(self) -> NDArray[np.integer[NBitBase]]:
        """Get the data channels for the MERFISH data set.

        Returns
            A list of the data channel indexes
        """
        return np.array(self.data.index)

    def get_data_colors(self, merfish_only=True):
        if merfish_only:
            return self.data[~self.data["bitNumber"].isna()]["color"].unique()
        return self.data["color"].unique()

    def get_channels_for_color(self, color):
        return self.data[self.data["color"] == color]["channelName"]

    def get_one_channel_per_round(self) -> NDArray[np.integer[NBitBase]]:
        """Get a list of data channels such that there is one (arbitrary) channel per imaging round."""
        rounds = self.data.groupby("imagingRound").first().channelName.to_numpy()
        return self.data[self.data["channelName"].isin(rounds)].index.to_numpy()

    def get_imaging_round_for_channel(self, data_channel: int) -> int:
        """Get the imaging round for a given data channel."""
        return self.data.iloc[data_channel]["imagingRound"]

    def get_data_channel_readout_name(self, data_channel: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            data_channel: The index of the data channel
        Returns:
            The name of the specified data channel
        """
        return self.data.iloc[data_channel]["readoutName"]

    def get_data_channel_name(self, data_channel: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            data_channel: The index of the data channel
        Returns:
            The name of the specified data channel,
            primarily relevant for non-multiplex measurements
        """
        return self.data.iloc[data_channel]["channelName"]

    def get_data_channel_index(self, data_channel_name: str) -> int:
        """Get the index for the data channel with the specified name.

        Args:
            data_channel_name: the name of the data channel. The data channel
                name is not case sensitive.

        Returns
            the index of the data channel where the data channel name is
                data_channel_name

        Raises
            # TODO this should raise a meaningful exception if the data channel
            # is not found
        """
        return self.data[
            self.data["channelName"].apply(lambda x: str(x).lower()) == data_channel_name.lower()
        ].index.to_numpy()[0]

    def get_data_channel_color(self, data_channel: int) -> str:
        """Get the color used for measuring the specified data channel.

        Args:
            data_channel: the data channel index
        Returns:
            the color for this data channel as a string
        """
        return str(self.data.loc[data_channel, "color"])

    def get_data_channel_for_bit(self, bit_name: str) -> int:
        """Get the data channel associated with the specified bit.

        Args:
            bit_name: the name of the bit to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data["readoutName"] == bit_name].index.values.item()

    def get_data_channel_with_name(self, channel_name: str) -> int:
        """Get the data channel associated with a gene name.

        Args:
            channel_name: the name of the gene to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data["channelName"] == channel_name].index.values.item()

    def get_fiducial_filename(self, data_channel: int, fov: str) -> str:
        """Get the path for the image file that contains the fiducial image for the specified dataChannel and fov.

        Args:
            data_channel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """
        image_type = self.data.loc[data_channel, "fiducialImageType"]
        imaging_round = self.data.loc[data_channel, "fiducialImagingRound"]
        return self._get_image_path(image_type, fov, imaging_round)

    def get_fiducial_frame_index(self, data_channel: int) -> int:
        """Get the index of the frame containing the fiducial image for the specified data channel.

        Args:
            data_channel: index of the data channel
        Returns:
            The index of the fiducial frame in the corresponding image file
        """
        return self.data.iloc[data_channel]["fiducialFrame"]

    def get_image_filename(self, data_channel: int, fov: str) -> str:
        """Get the path for the image file that contains the images for the specified dataChannel and fov.

        Args:
            data_channel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """
        channel_info = self.data.iloc[data_channel]
        return self._get_image_path(channel_info["imageType"], fov, channel_info["imagingRound"])

    def get_image_frame_index(self, data_channel: int, z_position: float) -> int:
        """Get the index of the frame containing the image for the specified data channel and z position.

        Args:
            data_channel: index of the data channel
            z_position: the z position
        Returns:
            The index of the frame in the corresponding image file
        """
        channel_info = self.data.iloc[data_channel]
        channel_z = channel_info["zPos"]
        if isinstance(channel_z, np.ndarray):
            z_index = np.where(channel_z == z_position)[0]
            if len(z_index) == 0:
                raise Exception(
                    "Requested z position not found. Position "
                    + "z=%0.2f not found for channel %i" % (z_position, data_channel)
                )
            frame_index = z_index[0]
        else:
            frame_index = 0

        frames = channel_info["frame"]
        return frames[frame_index] if isinstance(frames, np.ndarray) else frames

    def get_z_positions(self) -> np.ndarray:
        """Get the z positions present in this data organization.

        Returns
            A sorted list of all unique z positions
        """
        return np.sort(np.unique([y for x in self.data["zPos"] for y in x]))

    def get_fovs(self) -> np.ndarray:
        return np.unique(self.fileMap["fov"])

    def get_sequential_rounds(self) -> tuple[list[int], list[str]]:
        """Get the rounds that are not present in your codebook.

        Returns
            A tuple of two lists, the first list contains the channel number
            for all the rounds not contained in the codebook, the second list
            contains the name associated with that channel in the data
            organization file.
        """
        multiplex_bits = {b for x in self.dataset.get_codebooks() for b in x.get_bit_names()}
        sequential_channels = [
            i for i in self.get_data_channels() if self.get_data_channel_readout_name(i) not in multiplex_bits
        ]
        sequential_gene_names = [self.get_data_channel_name(x) for x in sequential_channels]
        return sequential_channels, sequential_gene_names

    def _get_image_path(self, image_type: str, fov: str, imaging_round: int) -> str:
        selection = self.fileMap[
            (self.fileMap["imageType"] == image_type)
            & (self.fileMap["fov"] == fov)
            & (self.fileMap["imagingRound"] == imaging_round)
        ]
        if selection.empty:
            selection = self.fileMap[
                (self.fileMap["imageType"] == image_type)
                & (self.fileMap["fov"].astype(int) == int(fov))
                & (self.fileMap["imagingRound"] == imaging_round)
            ]
        filemap_path = selection["imagePath"].to_numpy()[0]
        return self.dataset.raw_data_path / filemap_path

    def _truncate_file_path(self, path) -> None:
        head, tail = os.path.split(path)
        return tail

    def _map_image_files(self) -> None:
        # TODO: This doesn't map the fiducial image types and currently assumes
        # that the fiducial image types and regular expressions are part of the
        # standard image types.

        try:
            self.fileMap = self.dataset.load_dataframe_from_csv("filemap", dtype={"fov": str})
            if self.fov_list:
                self.fileMap = self.fileMap[self.fileMap["fov"].isin(self.fov_list)]
            if self.skip:
                self.fileMap = self.fileMap[~self.fileMap["fov"].isin(self.skip)]
        except FileNotFoundError as e:
            print("Mapping image files from data organization")
            unique_patterns = self.data.drop_duplicates(subset=["imageType", "imageRegExp"], keep="first")

            unique_types = unique_patterns["imageType"]
            unique_indices = unique_patterns.index.to_numpy()

            filenames = self.dataset.get_image_file_names()
            if len(filenames) == 0:
                raise dataset.DataFormatException("No image files found at %s." % self.dataset.rawDataPath) from e
            file_data = []
            for current_type, current_index in zip(unique_types, unique_indices):
                regex = re.compile(self.data.imageRegExp[current_index])

                matching_files = False
                for current_file in filenames:
                    matched_name = regex.search(current_file)
                    if matched_name is not None:
                        transformed_name = matched_name.groupdict()
                        if transformed_name["imageType"] == current_type:
                            if "imagingRound" not in transformed_name:
                                transformed_name["imagingRound"] = -1
                            transformed_name["imagePath"] = current_file
                            matching_files = True
                            file_data.append(transformed_name)

                if not matching_files:
                    raise dataset.DataFormatException(
                        "Unable to identify image files matching regular "
                        + "expression %s for image type %s." % (self.data.imageRegExp[current_index], current_type)
                    ) from e

            self.fileMap = pd.DataFrame(file_data)
            self.fileMap["imagingRound"] = self.fileMap["imagingRound"].astype(int)
            if "fov" not in self.fileMap:
                columns = sorted(self.fileMap.filter(like="fov").columns)
                self.fileMap["fov"] = self.fileMap[columns].agg("_".join, axis=1)

            if self.fov_list:
                self.fileMap = self.fileMap[self.fileMap["fov"].isin(self.fov_list)]

            if self.skip:
                self.fileMap = self.fileMap[~self.fileMap["fov"].isin(self.skip)]

            self._validate_file_map()

            self.dataset.save_dataframe_to_csv(self.fileMap, "filemap", index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _validate_file_map(self) -> None:
        """Ensure that all the files specified in the file map of the raw images are present.

        Raises
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """
        expected_image_size = None
        for data_channel in self.get_data_channels():
            for fov in self.get_fovs():
                channel_info = self.data.iloc[data_channel]
                try:
                    image_path = self._get_image_path(channel_info["imageType"], fov, channel_info["imagingRound"])
                except IndexError as e:
                    raise FileNotFoundError(
                        f"Unable to find image path for {channel_info['imageType']}, fov={fov}, round={channel_info['imagingRound']}"
                    ) from e

                if not self.dataset.raw_data_portal.open_file(image_path).exists():
                    raise InputDataError(
                        ("Image data for channel {0} and fov {1} not found. " "Expected at {2}").format(
                            data_channel, fov, image_path
                        )
                    )

                try:
                    image_size = self.dataset.image_stack_size(image_path)
                except Exception as exc:
                    raise InputDataError(
                        ("Unable to determine image stack size for fov {0} from" " data channel {1} at {2}").format(
                            data_channel, fov, image_path
                        )
                    ) from exc

                frames = channel_info["frame"]

                # this assumes fiducials are stored in the same image file
                if isinstance(channel_info["fiducialFrame"], np.ndarray):
                    required_frames = max(np.max(frames), np.max(channel_info["fiducialFrame"]))
                else:
                    required_frames = max(np.max(frames), channel_info["fiducialFrame"])
                if required_frames >= image_size[2]:
                    raise InputDataError(
                        (
                            "Insufficient frames in data for channel {0} and "
                            "fov {1}. Expected {2} frames "
                            "but only found {3} in file {4}"
                        ).format(data_channel, fov, required_frames, image_size[2], image_path)
                    )

                if expected_image_size is None:
                    expected_image_size = [image_size[0], image_size[1]]
                else:
                    if expected_image_size[0] != image_size[0] or expected_image_size[1] != image_size[1]:
                        raise InputDataError(
                            (
                                "Image data for channel {0} and fov {1} has "
                                "unexpected dimensions. Expected {2}x{3} but "
                                "found {4}x{5} in image file {6}"
                            ).format(
                                data_channel,
                                fov,
                                expected_image_size[0],
                                expected_image_size[1],
                                image_size[0],
                                image_size[1],
                                image_path,
                            )
                        )
