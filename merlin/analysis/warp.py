from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage import registration, transform
from sklearn.neighbors import NearestNeighbors

from merlin.core import analysistask
from merlin.util import aberration


class Warp(analysistask.AnalysisTask):

    """
    An abstract class for warping a set of images so that the corresponding
    pixels align between images taken in different imaging rounds.
    """

    def setup(self, *, parallel: bool) -> None:
        super().setup(parallel=parallel)

        self.set_default_parameters({"write_fiducial_images": False, "write_aligned_images": False})

        self.writeAlignedFiducialImages = self.parameters["write_fiducial_images"]

    def get_aligned_image_set(self, fov: int, chromaticCorrector: aberration.ChromaticCorrector = None) -> np.ndarray:
        """Get the set of transformed images for the specified fov.

        Args:
            fov: index of the field of view
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 4-dimensional numpy array containing the aligned images. The
                images are arranged as [channel, zIndex, x, y]
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zIndexes = range(len(self.dataSet.get_z_positions()))
        return np.array(
            [[self.get_aligned_image(fov, d, z, chromaticCorrector) for z in zIndexes] for d in dataChannels]
        )

    def get_aligned_image(
        self, fov: int, dataChannel: int, zIndex: int, chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        """Get the specified transformed image

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        inputImage = self.dataSet.get_raw_image(dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        transformation = self.get_transformation(dataChannel)
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization().get_data_channel_color(dataChannel)
            return transform.warp(
                chromaticCorrector.transform_image(inputImage, imageColor), transformation, preserve_range=True
            ).astype(inputImage.dtype)
        else:
            return transform.warp(inputImage, transformation, preserve_range=True).astype(inputImage.dtype)

    def _process_transformations(self, transformationList, fov) -> None:
        """
        Process the transformations determined for a given fov.

        The list of transformation is used to write registered images and
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel.
            fov: The fov that is being transformed.
        """

        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        if self.parameters["write_aligned_images"]:
            zPositions = self.dataSet.get_z_positions()

            imageDescription = self.dataSet.analysis_tiff_description(len(zPositions), len(dataChannels))

            with self.dataSet.writer_for_analysis_images(self, "aligned_images", fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    for z in zPositions:
                        inputImage = self.dataSet.get_raw_image(x, fov, z)
                        transformedImage = transform.warp(inputImage, t, preserve_range=True).astype(inputImage.dtype)
                        outputTif.save(transformedImage, photometric="MINISBLACK", metadata=imageDescription)

        if self.writeAlignedFiducialImages:
            fiducialImageDescription = self.dataSet.analysis_tiff_description(1, len(dataChannels))

            with self.dataSet.writer_for_analysis_images(self, "aligned_fiducial_images", fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    inputImage = self.dataSet.get_fiducial_image(x, fov)
                    transformedImage = transform.warp(inputImage, t, preserve_range=True).astype(inputImage.dtype)
                    outputTif.save(transformedImage, photometric="MINISBLACK", metadata=fiducialImageDescription)

        self._save_transformations(transformationList, fov)

    def _save_transformations(self, transformationList: List, fov: int) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array(transformationList, dtype=object),
            "offsets",
            self.analysis_name,
            resultIndex=fov,
            subdirectory="transformations",
        )

    def get_transformation(
        self, fov: int, dataChannel: int = None
    ) -> Union[transform.EuclideanTransform, List[transform.EuclideanTransform]]:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.
        Returns:
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformationMatrices = self.dataSet.load_numpy_analysis_result(
            "offsets", self, resultIndex=fov, subdirectory="transformations"
        )
        if dataChannel is not None:
            return transformationMatrices[
                self.dataSet.get_data_organization().get_imaging_round_for_channel(dataChannel) - 1
            ]
        else:
            return transformationMatrices


class FiducialCorrelationWarp(Warp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.set_default_parameters({"highpass_sigma": 3, "reference_round": 0})

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassSigma = self.parameters["highpass_sigma"]
        highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)

        return inputImage.astype(float) - cv2.GaussianBlur(
            inputImage, (highPassFilterSize, highPassFilterSize), highPassSigma, borderType=cv2.BORDER_REPLICATE
        )

    def run_analysis(self, fragment: str):
        fixedImage = self._filter(self.dataSet.get_fiducial_image(self.parameters["reference_round"], fragment))
        offsets = [
            registration.phase_cross_correlation(
                fixedImage, self._filter(self.dataSet.get_fiducial_image(x, fragment)), upsample_factor=100
            )[0]
            for x in self.dataSet.get_data_organization().get_one_channel_per_round()
        ]
        transformations = [transform.SimilarityTransform(translation=[-x[1], -x[0]]) for x in offsets]
        self._process_transformations(transformations, fragment)


class FiducialBeadWarp(Warp):
    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on alignment of local maxima (beads).
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.set_default_parameters(
            {
                "delta": 2,
                "delta_fit": 3,
                "dbscan": True,
                "max_disp": 200,
                "tile_size": 512,
                "filter_size": 15,
                "threshold_sigma": 4,
                "reference_round": 0,
            }
        )

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        im = inputImage.astype(np.float32)
        s = self.parameters["filter_size"]
        return im / cv2.blur(im, (s, s))

    def _get_tiles(self, im):
        size = self.parameters["tile_size"]
        sx, sy = im.shape
        Mx = int(np.ceil(sx / float(size)))
        My = int(np.ceil(sy / float(size)))
        ims_dic = {}
        for ix in range(Mx):
            for iy in range(My):
                ims_dic[(ix, iy)] = ims_dic.get((ix, iy), []) + [
                    im[ix * size : (ix + 1) * size, iy * size : (iy + 1) * size]
                ]
        return ims_dic

    def _get_local_max(self, im_dif, th_fit):
        """Given a 2D image <im_dif> as numpy array, get the local maxima in square -<delta>_to_<delta>.
        Optional a dbscan can be used to couple connected pixels with the same local maximum.
        (This is important if saturating the camera values.)
        Returns: Xh - a list of x,y and brightness of the local maxima
        """
        delta = self.parameters["delta"]
        delta_fit = self.parameters["delta_fit"]
        x, y = np.where(im_dif > th_fit)
        xmax, ymax = im_dif.shape
        in_im = im_dif[x, y]
        keep = np.ones(len(x)) > 0
        for d2 in range(-delta, delta + 1):
            for d3 in range(-delta, delta + 1):
                keep &= in_im >= im_dif[(x + d2) % xmax, (y + d3) % ymax]
        x, y = x[keep], y[keep]
        h = in_im[keep]
        Xh = np.array([x, y, h]).T
        if self.parameters["dbscan"] and len(Xh) > 0:
            from scipy import ndimage

            im_keep = np.zeros(im_dif.shape, dtype=bool)
            im_keep[x, y] = True
            lbl, nlbl = ndimage.label(im_keep, structure=np.ones([3] * 2))
            l = lbl[x, y]  # labels after reconnection
            ul = np.arange(1, nlbl + 1)
            il = np.argsort(l)
            l = l[il]
            x, y, h = x[il], y[il], h[il]
            inds = np.searchsorted(l, ul)
            Xh = np.array([x, y, h]).T
            Xh_ = []
            for i_ in range(len(inds)):
                j_ = inds[i_ + 1] if i_ < len(inds) - 1 else len(Xh)
                Xh_.append(np.mean(Xh[inds[i_] : j_], 0))
            Xh = np.array(Xh_)
            x, y, h = Xh.T
        im_centers = []
        if delta_fit != 0 and len(Xh) > 0:
            x, y, h = Xh.T
            x, y = x.astype(int), y.astype(int)
            im_centers = [[], [], []]
            for d2 in range(-delta_fit, delta_fit + 1):
                for d3 in range(-delta_fit, delta_fit + 1):
                    if (d2 * d2 + d3 * d3) <= (delta_fit * delta_fit):
                        im_centers[0].append((x + d2))
                        im_centers[1].append((y + d3))
                        im_centers[2].append(im_dif[(x + d2) % xmax, (y + d3) % ymax])
            im_centers_ = np.array(im_centers)
            im_centers_[-1] -= np.min(im_centers_[-1], axis=0)
            xc = np.sum(im_centers_[0] * im_centers_[-1], axis=0) / np.sum(im_centers_[-1], axis=0)
            yc = np.sum(im_centers_[1] * im_centers_[-1], axis=0) / np.sum(im_centers_[-1], axis=0)
            Xh = np.array([xc, yc, h]).T

        return Xh

    def _fftalign_2d(self, im1, im2, center=[0, 0]):
        """
        Inputs: 2 2D images <im1>, <im2>, the expected displacement <center>,
            the maximum displacement <max_disp> around the expected vector.
        This computes the cross-cor between im1 and im2 using fftconvolve (fast) and determines the maximum
        """
        from scipy.signal import fftconvolve

        im2_ = np.array(im2[::-1, ::-1], dtype=float)
        im2_ -= np.mean(im2_)
        im2_ /= np.std(im2_)
        im1_ = np.array(im1, dtype=float)
        im1_ -= np.mean(im1_)
        im1_ /= np.std(im1_)
        im_cor = fftconvolve(im1_, im2_, mode="full")

        sx_cor, sy_cor = im_cor.shape
        center_ = np.array(center) + np.array([sx_cor, sy_cor]) / 2.0

        max_disp = self.parameters["max_disp"]
        x_min = int(min(max(center_[0] - max_disp, 0), sx_cor))
        x_max = int(min(max(center_[0] + max_disp, 0), sx_cor))
        y_min = int(min(max(center_[1] - max_disp, 0), sy_cor))
        y_max = int(min(max(center_[1] + max_disp, 0), sy_cor))

        im_cor0 = np.zeros_like(im_cor)
        im_cor0[x_min:x_max, y_min:y_max] = 1
        im_cor = im_cor * im_cor0

        y, x = np.unravel_index(np.argmax(im_cor), im_cor.shape)
        if np.sum(im_cor > 0) > 0:
            im_cor[im_cor == 0] = np.min(im_cor[im_cor > 0])

        return (-(np.array(im_cor.shape) - 1) / 2.0 + [y, x]).astype(int)

    def run_analysis(self, fragment: str):
        fixedImage = self._filter(self.dataSet.get_fiducial_image(self.parameters["reference_round"], fragment))
        offsets = []
        im2 = fixedImage.copy()
        for channel in self.dataSet.get_data_organization().get_one_channel_per_round():
            im_beads = self._filter(self.dataSet.get_fiducial_image(channel, fragment))
            im1 = im_beads.copy()
            Txyzs = []
            dic_ims1 = self._get_tiles(im1)
            dic_ims2 = self._get_tiles(im2)
            for key in dic_ims1:
                im1_ = dic_ims1[key][0]
                im2_ = dic_ims2[key][0]

                Xh1 = self._get_local_max(im1_, np.mean(im1_) + np.std(im1_) * self.parameters["threshold_sigma"])
                Xh2 = self._get_local_max(im2_, np.mean(im2_) + np.std(im2_) * self.parameters["threshold_sigma"])

                if len(Xh1) > 0 and len(Xh2) > 0:
                    tx, ty = self._fftalign_2d(im1_, im2_, center=[0, 0])
                    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(Xh1[:, :2])
                    distances, indices = nbrs.kneighbors(Xh2[:, :2] + [tx, ty])
                    keep = distances.flatten() < 3
                    indices_ = indices.flatten()[keep]
                    if len(indices_) > 0:
                        try:
                            Txyz = np.median(Xh2[keep, :2] - Xh1[indices_, :2], 0)
                        except Exception:
                            print("Xh1", Xh1)
                            print("Xh2", Xh2)
                            print("keep", keep)
                            print("indices_", indices_)
                            raise
                        Txyzs.append(Txyz)
                    else:
                        pass
                        # print("No kept beads, fragmentIndex", fragmentIndex, ", channel", channel, ", tile", key)
                else:
                    print("No beads found, fragmentIndex", fragment, ", channel", channel, ", tile", key)
            offsets.append(np.median(Txyzs, 0))
        transformations = [transform.SimilarityTransform(translation=[-x[1], -x[0]]) for x in offsets]
        self._process_transformations(transformations, fragment)


class FiducialAlign(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.set_default_parameters({"sz_norm": 20, "sz": 500, "nelems": 7, "reference_round": 0})

        self.define_results("drifts", "tile_drifts")

    def get_aligned_image_set(self, fov: int, chromaticCorrector: aberration.ChromaticCorrector = None) -> np.ndarray:
        """Get the set of transformed images for the specified fov.

        Args:
            fov: index of the field of view
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.

        Returns
            a 4-dimensional numpy array containing the aligned images. The
                images are arranged as [channel, zIndex, x, y]

        """
        channels = self.dataSet.get_data_organization().get_data_channels()
        z_indexes = range(len(self.dataSet.get_z_positions()))
        return np.array([[self.get_aligned_image(fov, d, z, chromaticCorrector) for z in z_indexes] for d in channels])

    def get_aligned_image(
        self, fov: str, channel: int, z_index: int, chromatic_corrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        """Get the specified transformed image.

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.

        Returns
            a 2-dimensional numpy array containing the specified image
        """
        try:
            zdrift, xdrift, ydrift = self.get_transformation(channel)
        except ValueError:  # Drift correction was not 3D
            zdrift = 0
            xdrift, ydrift = self.get_transformation(channel)
        try:
            input_image = self.dataSet.get_raw_image(
                channel,
                fov,
                self.dataSet.z_index_to_position(z_index - zdrift),
            )
            if chromatic_corrector is not None:
                image_color = self.dataSet.get_data_organization().get_data_channel_color(channel)
                input_image = chromatic_corrector.transform_image(input_image, image_color).astype(input_image.dtype)
        except IndexError:  # Z drift outside bounds
            input_image = self.dataSet.get_raw_image(channel, fov, self.dataSet.z_index_to_position(0))
            return np.zeros_like(input_image)
        else:
            return ndimage.shift(input_image, [-xdrift, -ydrift], order=0)

    def get_transformation(self, channel: int = None) -> np.ndarray:
        """Get the transformations for aligning images for the specified field of view."""
        drifts = self.load_result("drifts")
        if channel is None:
            return drifts
        lowest = np.inf
        for c in self.dataSet.get_data_organization().get_data_channels():
            imaging_round = self.dataSet.get_data_organization().get_imaging_round_for_channel(c)
            if imaging_round > 0 and imaging_round < lowest:
                lowest = imaging_round
        lowest -= 1
        return drifts[self.dataSet.get_data_organization().get_imaging_round_for_channel(channel) - lowest]

    def get_tiles(self, image: np.ndarray, size: int = 256) -> list[np.ndarray]:
        """Split the image into tiles and return them as a list."""
        tiles = np.ceil(np.array(image.shape) / size)
        res = [image]
        for axis, n in enumerate(tiles):
            res = sum((np.array_split(im, n, axis) for im in res), [])
        return res

    def norm_slice(self, image: np.ndarray, sigma: int) -> np.ndarray:
        """Perform a high-pass filter on the image."""
        if image.ndim == 3:
            return np.array([zslice - cv2.blur(zslice, (sigma, sigma)) for zslice in image])
        return image - cv2.blur(image, (sigma, sigma))

    def get_txyz(self, moving_image: np.ndarray) -> Tuple[np.ndarray, list[np.ndarray]]:
        moving_image = self.norm_slice(moving_image.astype(np.float32), self.parameters["sz_norm"])
        moving_tiles = self.get_tiles(moving_image, size=self.parameters["sz"])
        txyzs = []
        for ib in self.tiles_to_align:
            im0 = self.fixed_tiles[ib]
            im1 = moving_tiles[ib]
            im0 = (im0 - np.mean(im0)) / np.std(im0)
            im1 = (im1 - np.mean(im1)) / np.std(im1)
            if im0.ndim == 3:
                im_cor = fftconvolve(im0[::-1, ::-1, ::-1], im1, mode="full")
            else:
                im_cor = fftconvolve(im0[::-1, ::-1], im1, mode="full")
            txyz = np.unravel_index(np.argmax(im_cor), im_cor.shape) - np.array(im0.shape) + 1
            txyzs.append(txyz)
        txyz = np.median(txyzs, 0).astype(int)
        return txyz, txyzs

    def run_analysis(self) -> None:
        """
        Given two 3D images im_dapi0,im_dapi1, this normalizes them by subtracting local background
        and then computes correlations on <nelemes> blocks with highest  std of signal of size sz
        It will return median value and a list of single values.
        """
        fixed_image = self.dataSet.get_fiducial_image(self.parameters["reference_round"], self.fragment)
        fixed_image = self.norm_slice(fixed_image.astype(np.float32), self.parameters["sz_norm"])
        self.fixed_tiles = self.get_tiles(fixed_image, size=self.parameters["sz"])
        best = np.argsort([np.std(tile) for tile in self.fixed_tiles])[::-1]
        self.tiles_to_align = best[: min(self.parameters["nelems"], len(best))]
        drifts = []
        tile_drifts = []
        for channel in self.dataSet.get_data_organization().get_one_channel_per_round():
            moving_image = self.dataSet.get_fiducial_image(channel, self.fragment)
            txyz, txyzs = self.get_txyz(moving_image)
            imaging_round = self.dataSet.get_data_organization().get_imaging_round_for_channel(channel)
            drifts.append((imaging_round, txyz))
            tile_drifts.append((imaging_round, txyzs))
        self.drifts = np.array([x[1] for x in sorted(drifts)])
        self.tile_drifts = np.array([x[1] for x in sorted(tile_drifts)])

    def metadata(self) -> dict:
        return {f"drift_{i}": drifts for i, drifts in enumerate(self.drifts)}
