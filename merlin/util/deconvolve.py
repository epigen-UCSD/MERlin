import cv2
import numpy as np
import torch
from scipy import ndimage
from sdeconv.deconv import SWiener
from sdeconv.core import SSettings

from merlin.util import matlab

"""
This module containts utility functions for performing deconvolution on
images.
"""


def calculate_projectors(windowSize: int, sigmaG: float) -> list:
    """Calculate forward and backward projectors as described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function

    Returns:
        A list containing the forward and backward projectors to use for
        Lucy-Richardson deconvolution.
    """
    pf = matlab.matlab_gauss2D(shape=(windowSize, windowSize), sigma=sigmaG)
    pfFFT = np.fft.fft2(pf)

    # Wiener-Butterworth back projector.
    #
    # These values are from Guo et al.
    alpha = 0.001
    beta = 0.001
    n = 8

    # This is the cut-off frequency.
    kc = 1.0 / (0.5 * 2.355 * sigmaG)

    # FFT frequencies
    kv = np.fft.fftfreq(pfFFT.shape[0])

    kx = np.zeros((kv.size, kv.size))
    for i in range(kv.size):
        kx[i, :] = np.copy(kv)

    ky = np.transpose(kx)
    kk = np.sqrt(kx * kx + ky * ky)

    # Wiener filter
    bWiener = pfFFT / (np.abs(pfFFT) * np.abs(pfFFT) + alpha)

    # Buttersworth filter
    eps = np.sqrt(1.0 / (beta * beta) - 1)

    kkSqr = kk * kk / (kc * kc)
    bBWorth = 1.0 / np.sqrt(1.0 + eps * eps * np.power(kkSqr, n))

    # Weiner-Butterworth back projector
    pbFFT = bWiener * bBWorth

    # back projector.
    pb = np.real(np.fft.ifft2(pbFFT))

    return [pf, pb]


def deconvolve_lucyrichardson(image: np.ndarray, windowSize: int, sigmaG: float, iterationCount: int) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function.

    Ported from Matlab deconvlucy.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform

    Returns:
        the deconvolved image
    """
    eps = np.finfo(float).eps
    Y = np.copy(image)
    J1 = np.copy(image)
    J2 = np.copy(image)
    wI = np.copy(image)
    imR = np.copy(image)
    reblurred = np.copy(image)
    tmpMat1 = np.zeros(image.shape, dtype=float)
    tmpMat2 = np.zeros(image.shape, dtype=float)
    T1 = np.zeros(image.shape, dtype=float)
    T2 = np.zeros(image.shape, dtype=float)
    l = 0

    if windowSize % 2 != 1:
        gaussianFilter = matlab.matlab_gauss2D(shape=(windowSize, windowSize), sigma=sigmaG)

    for i in range(iterationCount):
        if i > 1:
            cv2.multiply(T1, T2, tmpMat1)
            cv2.multiply(T2, T2, tmpMat2)
            l = np.sum(tmpMat1) / (np.sum(tmpMat2) + eps)
            l = max(min(l, 1), 0)
        cv2.subtract(J1, J2, Y)
        cv2.addWeighted(J1, 1, Y, l, 0, Y)
        np.clip(Y, 0, None, Y)
        if windowSize % 2 == 1:
            cv2.GaussianBlur(Y, (windowSize, windowSize), sigmaG, reblurred, borderType=cv2.BORDER_REPLICATE)
        else:
            reblurred = ndimage.convolve(Y, gaussianFilter, mode="constant")
        np.clip(reblurred, eps, None, reblurred)
        cv2.divide(wI, reblurred, imR)
        imR += eps
        if windowSize % 2 == 1:
            cv2.GaussianBlur(imR, (windowSize, windowSize), sigmaG, imR, borderType=cv2.BORDER_REPLICATE)
        else:
            imR = ndimage.convolve(imR, gaussianFilter, mode="constant")
            imR[imR > 2**16] = 0
        np.copyto(J2, J1)
        np.multiply(Y, imR, out=J1)
        np.copyto(T2, T1)
        np.subtract(J1, Y, out=T1)
    return J1


def deconvolve_lucyrichardson_guo(
    image: np.ndarray, window_size: int, sigma: float, iterations: int, *, gpu: bool = False
) -> np.ndarray:
    """Performs Lucy-Richardson deconvolution on the provided image using a
    Gaussian point spread function. This version used the optimized
    deconvolution approach described in:

    'Accelerating iterative deconvolution and multiview fusion by orders
    of magnitude', Guo et al, bioRxiv 2019.

    Args:
        image: the input image to be deconvolved
        windowSize: the size of the window over which to perform the gaussian.
            This must be an odd number.
        sigmaG: the standard deviation of the Gaussian point spread function
        iterationCount: the number of iterations to perform
        gpu: Whether to perform operations on the GPU

    Returns:
        the deconvolved image
    """
    [pf, pb] = calculate_projectors(window_size, sigma)

    eps = 1.0e-6
    i_max = 2**16 - 1

    ek = np.copy(image)
    np.clip(ek, eps, None, ek)

    for _ in range(iterations):
        ekf = cv2.filter2D(ek, -1, pf, borderType=cv2.BORDER_REPLICATE)
        np.clip(ekf, eps, i_max, ekf)

        ek = ek * cv2.filter2D(image / ekf, -1, pb, borderType=cv2.BORDER_REPLICATE)
        np.clip(ek, eps, i_max, ek)

    return ek


def prepare_psf(psf, shape):
    psff = np.zeros(shape, dtype=np.float32)

    slices = [
        (
            (slice((s_psff - s_psf_full_) // 2, (s_psff + s_psf_full_) // 2), slice(None))
            if s_psff > s_psf_full_
            else (slice(None), slice((s_psf_full_ - s_psff) // 2, (s_psf_full_ + s_psff) // 2))
        )
        for s_psff, s_psf_full_ in zip(psff.shape, psf.shape)
    ]
    sl_psff, sl_psf_full_ = list(zip(*slices))
    psff[sl_psff] = psf[sl_psf_full_]
    return psff


def deconvolve_sdeconv(im, psf, device="cpu"):
    obj = SSettings.instance()
    obj.device = device
    if psf.shape != im.shape:
        psf = prepare_psf(psf, im.shape)

    pad = int(np.min(list(np.array(im.shape) - 1) + [50]))
    filter_ = SWiener(torch.from_numpy(psf).to(device), beta=0.005, pad=pad)
    return filter_(torch.from_numpy(im).to(device)).cpu().detach().numpy().astype(np.float32)


def deconvolve_tiles(image, psf, device="cpu", tile_size=500, pad=100):
    im0 = np.zeros_like(image)
    sx, sy = image.shape[1:]
    ixys = []
    for ix in np.arange(0, sx, tile_size):
        for iy in np.arange(0, sy, tile_size):
            ixys.append([ix, iy])

    for ix, iy in ixys:
        imsm = image[:, ix : ix + pad + tile_size, iy : iy + pad + tile_size]
        imt = deconvolve_sdeconv(imsm, psf, device)
        torch.cuda.empty_cache()
        start_x = ix + pad // 2 if ix > 0 else 0
        end_x = ix + pad // 2 + tile_size
        start_y = iy + pad // 2 if iy > 0 else 0
        end_y = iy + pad // 2 + tile_size
        im0[:, start_x:end_x, start_y:end_y] = imt[:, (start_x - ix) : (end_x - ix), (start_y - iy) : (end_y - iy)]
    return im0
