from typing import Tuple

import cv2
import numba
import numpy as np
from skimage import measure

from merlin.data import codebook as mcodebook
from merlin.util import binary

"""
Utility functions for pixel based decoding.
"""


def normalize(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    else:
        return x


@numba.njit
def sum_labels(labels, weights):
    return np.bincount(labels.ravel(), weights=weights.ravel())[1:]


@numba.njit
def label_image_2d(decoded_image):
    xdim, ydim = decoded_image.shape
    labels = np.zeros_like(decoded_image, dtype=np.uint32)
    current_label = 0
    barcode_index = []
    stack = []
    for idx in zip(*np.where(decoded_image >= 0)):
        if labels[idx] == 0:
            current_label += 1
            barcode_index.append(decoded_image[idx])
            stack.append(idx)
            while stack:
                x, y = stack.pop()
                if decoded_image[x, y] != decoded_image[idx] or labels[x, y] != 0:
                    continue  # already visited or not part of this group
                labels[x, y] = current_label
                if x > 0:
                    stack.append((x - 1, y))
                if x + 1 < xdim:
                    stack.append((x + 1, y))
                if y > 0:
                    stack.append((x, y - 1))
                if y + 1 < ydim:
                    stack.append((x, y + 1))
    return labels, current_label, barcode_index


@numba.njit
def label_image_3d(decoded_image):
    zdim, xdim, ydim = decoded_image.shape
    labels = np.zeros_like(decoded_image, dtype=np.uint32)
    current_label = 0
    barcode_index = []
    stack = []
    for idx in zip(*np.where(decoded_image >= 0)):
        if labels[idx] == 0:
            current_label += 1
            barcode_index.append(decoded_image[idx])
            stack.append(idx)
            while stack:
                z, x, y = stack.pop()
                if decoded_image[z, x, y] != decoded_image[idx] or labels[z, x, y] != 0:
                    continue  # already visited or not part of this group
                labels[z, x, y] = current_label
                if z > 0:
                    stack.append((z - 1, x, y))
                if z + 1 < zdim:
                    stack.append((z + 1, x, y))
                if x > 0:
                    stack.append((z, x - 1, y))
                if x + 1 < xdim:
                    stack.append((z, x + 1, y))
                if y > 0:
                    stack.append((z, x, y - 1))
                if y + 1 < ydim:
                    stack.append((z, x, y + 1))
    return labels, current_label, barcode_index


class PixelBasedDecoder(object):
    def __init__(self, codebook: mcodebook.Codebook, scaleFactors: np.ndarray = None, backgrounds: np.ndarray = None):
        self._codebook = codebook
        self._decodingMatrix = self._calculate_normalized_barcodes()
        self._barcodeCount = self._decodingMatrix.shape[0]
        self._bitCount = self._decodingMatrix.shape[1]

        if scaleFactors is None:
            self._scaleFactors = np.ones(self._decodingMatrix.shape[1])
        else:
            self._scaleFactors = scaleFactors.copy()

        if backgrounds is None:
            self._backgrounds = np.zeros(self._decodingMatrix.shape[1])
        else:
            self._backgrounds = backgrounds.copy()

        self.refactorAreaThreshold = 4

    def decode_pixels(
        self,
        image: np.ndarray,
        scaleFactors: np.ndarray = None,
        backgrounds: np.ndarray = None,
        distanceThreshold: float = 0.5176,
        magnitudeThreshold: float = 1,
        lowPassSigma: float = 1,
    ):
        """Assign barcodes to the pixels in the provided image stock.

        Each pixel is assigned to the nearest barcode from the codebook if
        the distance between the normalized pixel trace and the barcode is
        less than the distance threshold.

        Args:
            imageData: input image stack. The first dimension indexes the bit
                number and the second and third dimensions contain the
                corresponding image.
            scaleFactors: factors to rescale each bit prior to normalization.
                The length of scaleFactors must be equal to the number of bits.
            backgrounds: background to subtract from each bit prior to applying
                the scale factors and prior to normalization. The length of
                backgrounds must be equal to the number of bits.
            distanceThreshold: the maximum distance between an assigned pixel
                and the nearest barcode. Pixels for which the nearest barcode
                is greater than distanceThreshold are left unassigned.
            magnitudeThreshold: the minimum pixel magnitude for which a
                barcode can be assigned that pixel. All pixels that fall
                below the magnitude threshold are not assigned a barcode
                in the decoded image.
            lowPassSigma: standard deviation for the low pass filter that is
                applied to the images prior to decoding.
        Returns:
            Four results are returned as a tuple (decodedImage, pixelMagnitudes,
                normalizedPixelTraces, distances). decodedImage is an image
                indicating the barcode index assigned to each pixel. Pixels
                for which a barcode is not assigned have a value of -1.
                pixelMagnitudes is an image where each pixel is the norm of
                the pixel trace after scaling by the provided scaleFactors.
                normalizedPixelTraces is an image stack containing the
                normalized intensities for each pixel. distances is an
                image containing the distance for each pixel to the assigned
                barcode.
        """
        if scaleFactors is None:
            scaleFactors = self._scaleFactors
        if backgrounds is None:
            backgrounds = self._backgrounds

        image_shape = image.shape
        image_data = np.empty(image_shape, dtype=np.float32)
        filter_size = int(2 * np.ceil(2 * lowPassSigma) + 1)
        for i in range(image_shape[0]):
            image_data[i, :, :] = cv2.GaussianBlur(image[i, :, :], (filter_size, filter_size), lowPassSigma)

        image_data = np.reshape(image_data, (image_shape[0], np.prod(image_shape[1:])))
        image_data = (image_data.T - backgrounds) / scaleFactors

        pixel_magnitudes = np.array(np.linalg.norm(image_data, axis=1), dtype=np.float32)
        pixel_magnitudes[pixel_magnitudes == 0] = 1

        image_data = image_data / pixel_magnitudes[:, None]

        dot = np.dot(image_data, self._decodingMatrix.T)
        indexes = np.argmax(dot, axis=1)
        distances = np.sqrt(2 * (1 - dot[np.arange(dot.shape[0]), indexes]))

        indexes[distances > distanceThreshold] = -1
        decoded_image = np.reshape(indexes, image_shape[1:])

        pixel_magnitudes = np.reshape(pixel_magnitudes, image_shape[1:])
        image_data = np.moveaxis(image_data, 1, 0)
        image_data = np.reshape(image_data, image_shape)
        distances = np.reshape(distances, image_shape[1:])

        decoded_image[pixel_magnitudes < magnitudeThreshold] = -1
        return decoded_image, pixel_magnitudes, image_data, distances

    def extract_all_barcodes(
        self,
        decodedImage: np.ndarray,
        pixelMagnitudes: np.ndarray,
        pixelTraces: np.ndarray,
        distances: np.ndarray,
        fov: int,
        cropWidth: int,
        zIndex: int = None,
        globalAligner=None,
        minimumArea: int = 0,
    ):
        """Extract the barcode information from the decoded image for barcodes
        that were decoded to the specified barcode index.

        Args:
            barcodeIndex: the index of the barcode to extract the corresponding
                barcodes
            decodedImage: the image indicating the barcode index assigned to
                each pixel
            pixelMagnitudes: an image containing norm of the intensities for
                each pixel across all bits after scaling by the scale factors
            pixelTraces: an image stack containing the normalized pixel
                intensity traces
            distances: an image indicating the distance between the normalized
                pixel trace and the assigned barcode for each pixel
            fov: the index of the field of view
            cropWidth: the number of pixels around the edge of each image within
                which barcodes are excluded from the output list.
            zIndex: the index of the z position
            globalAligner: the aligner used for converted to local x,y
                coordinates to global x,y coordinates
            minimumArea: the minimum area of barcodes to identify. Barcodes
                less than the specified minimum area are ignored.
        Returns:
            a pandas dataframe containing all the barcodes decoded with the
                specified barcode index
        """
        is3D = len(pixelTraces.shape) == 4
        if is3D:
            labels, nlabels, barcode_index = label_image_3d(decodedImage)
        else:
            labels, nlabels, barcode_index = label_image_2d(decodedImage)

        nbits = pixelTraces.shape[1] if is3D else pixelTraces.shape[0]

        if nlabels == 0:
            return np.array([], dtype=np.float32).reshape((0, 12 + nbits))

        result = np.empty((nlabels, 12 + nbits), dtype=np.float32)
        # Area
        result[:, 2] = np.bincount(labels.ravel())[1:]
        # Mean intensity
        result[:, 0] = sum_labels(labels, pixelMagnitudes) / result[:, 2]
        # Max intensity
        order = pixelMagnitudes[labels > 0].ravel().argsort()
        maxs = np.zeros(labels.max() + 1, pixelMagnitudes.dtype)
        maxs[labels[labels > 0].ravel()[order]] = pixelMagnitudes[labels > 0].ravel()[order]
        result[:, 1] = maxs[1:]
        # Mean distance
        result[:, 3] = np.bincount(labels.ravel(), weights=distances.ravel())[1:] / result[:, 2]
        # Min distance
        order = distances[labels > 0].ravel().argsort()
        mins = np.zeros(labels.max() + 1, distances.dtype)
        mins[labels[labels > 0].ravel()[order][::-1]] = distances[labels > 0].ravel()[order][::-1]
        result[:, 4] = mins[1:]

        # Centroids
        normalizer = sum_labels(labels, pixelMagnitudes)
        grids = np.ogrid[[slice(0, i) for i in pixelMagnitudes.shape]]
        if is3D:
            result[:, [7, 6, 5]] = np.array(
                [
                    sum_labels(labels, pixelMagnitudes * grids[dim].astype(float)) / normalizer
                    for dim in range(pixelMagnitudes.ndim)
                ]
            ).T
        else:
            result[:, [6, 5]] = np.array(
                [
                    sum_labels(labels, pixelMagnitudes * grids[dim].astype(float)) / normalizer
                    for dim in range(pixelMagnitudes.ndim)
                ]
            ).T
            result[:, 7] = zIndex

        # Global centroid coordinates
        if globalAligner is not None:
            result[:, [10, 8, 9]] = globalAligner.fov_coordinate_array_to_global(fov, result[:, [7, 5, 6]])
        else:
            result[:, [8, 9, 10]] = result[:, [5, 6, 7]]

        # Per-bit intensity
        for i, bit in enumerate(range(nbits), start=11):
            if is3D:
                result[:, i] = sum_labels(labels, pixelTraces[:, bit, :, :]) / result[:, 2]
            else:
                result[:, i] = sum_labels(labels, pixelTraces[bit, :, :]) / result[:, 2]

        result[:, -1] = barcode_index

        result = result[
            (result[:, 5] >= cropWidth)
            & (result[:, 5] < decodedImage.shape[-2])
            & (result[:, 6] >= cropWidth)
            & (result[:, 6] < decodedImage.shape[-1])
            & (result[:, 2] >= minimumArea)
        ]

        return result

    def _calculate_normalized_barcodes(self, ignoreBlanks=False, includeErrors=False):
        """Normalize the barcodes present in the provided codebook so that
        their L2 norm is 1.

        Args:
            ignoreBlanks: Flag to set if the barcodes corresponding to blanks
                should be ignored. If True, barcodes corresponding to a name
                that contains 'Blank' are ignored.
            includeErrors: Flag to set if barcodes corresponding to single bit
                errors should be added.
        Returns:
            A 2d numpy array where each row is a normalized barcode and each
                column is the corresponding normalized bit value.
        """

        barcodeSet = self._codebook.get_barcodes(ignoreBlanks=ignoreBlanks)
        magnitudes = np.sqrt(np.sum(barcodeSet * barcodeSet, axis=1))

        if not includeErrors:
            weightedBarcodes = np.array([normalize(x) for x, m in zip(barcodeSet, magnitudes)])
            return weightedBarcodes.astype(np.float32)

        else:
            barcodesWithSingleErrors = []
            for b in barcodeSet:
                barcodeSet = np.array([b] + [binary.flip_bit(b, i) for i in range(len(b))])
                bcMagnitudes = np.sqrt(np.sum(barcodeSet * barcodeSet, axis=1))
                weightedBC = np.array([x / m for x, m in zip(barcodeSet, bcMagnitudes)])
                barcodesWithSingleErrors.append(weightedBC)
            return np.array(barcodesWithSingleErrors, dtype=np.float32)

    def extract_refactors(
        self, decodedImage, pixelMagnitudes, normalizedPixelTraces, extractBackgrounds=False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the scale factors that would result in the mean
        on bit intensity for each bit to be equal.

        This code follows the legacy matlab decoder.

        If the scale factors for this decoder are not set to 1, then the
        calculated scale factors are dependent on the input scale factors
        used for the decoding.

        Args:
            imageSet: the image stack to decode in order to determine the
                scale factors
        Returns:
             a tuple containing an array of the scale factors, an array
                of the backgrounds, and an array of the abundance of each
                barcode determined during the decoding. For the scale factors
                and the backgrounds, the i'th entry is the scale factor
                for bit i. If extractBackgrounds is false, the returned
                background array is all zeros.
        """

        if extractBackgrounds:
            backgroundRefactors = self._extract_backgrounds(decodedImage, pixelMagnitudes, normalizedPixelTraces)
        else:
            backgroundRefactors = np.zeros(self._bitCount)

        sumPixelTraces = np.zeros((self._barcodeCount, self._bitCount))
        barcodesSeen = np.zeros(self._barcodeCount)
        for b in range(self._barcodeCount):
            barcodeRegions = [
                x
                for x in measure.regionprops(measure.label((decodedImage == b).astype(int)))
                if x.area >= self.refactorAreaThreshold
            ]
            barcodesSeen[b] = len(barcodeRegions)
            for br in barcodeRegions:
                meanPixelTrace = (
                    np.mean(
                        [normalizedPixelTraces[:, y[0], y[1]] * pixelMagnitudes[y[0], y[1]] for y in br.coords], axis=0
                    )
                    - backgroundRefactors
                )
                normPixelTrace = meanPixelTrace / np.linalg.norm(meanPixelTrace)
                sumPixelTraces[b, :] += normPixelTrace / barcodesSeen[b]

        sumPixelTraces[self._decodingMatrix == 0] = np.nan
        onBitIntensity = np.nanmean(sumPixelTraces, axis=0)
        refactors = onBitIntensity / np.mean(onBitIntensity)

        return refactors.astype(np.float32), backgroundRefactors.astype(np.float32), barcodesSeen

    def _extract_backgrounds(self, decodedImage, pixelMagnitudes, normalizedPixelTraces) -> np.ndarray:
        """Calculate the backgrounds to be subtracted for the the mean off
        bit intensity for each bit to be equal to zero.

        Args:
            imageSet: the image stack to decode in order to determine the
                scale factors
        Returns:
            an array of the backgrounds where the i'th entry is the scale factor
                for bit i.
        """
        sumMinPixelTraces = np.zeros((self._barcodeCount, self._bitCount))
        barcodesSeen = np.zeros(self._barcodeCount)
        # TODO this core functionality is very similar to that above. They
        # can be abstracted
        for b in range(self._barcodeCount):
            barcodeRegions = [
                x for x in measure.regionprops(measure.label((decodedImage == b).astype(int))) if x.area >= 5
            ]
            barcodesSeen[b] = len(barcodeRegions)
            for br in barcodeRegions:
                minPixelTrace = np.min(
                    [normalizedPixelTraces[:, y[0], y[1]] * pixelMagnitudes[y[0], y[1]] for y in br.coords], axis=0
                )
                sumMinPixelTraces[b, :] += minPixelTrace

        offPixelTraces = sumMinPixelTraces.copy()
        offPixelTraces[self._decodingMatrix > 0] = np.nan
        offBitIntensity = np.nansum(offPixelTraces, axis=0) / np.sum(
            (self._decodingMatrix == 0) * barcodesSeen[:, np.newaxis], axis=0
        )
        backgroundRefactors = offBitIntensity

        return backgroundRefactors
