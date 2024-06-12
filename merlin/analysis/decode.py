import os
import tempfile

import cv2
import numpy as np
from scipy.spatial import cKDTree

from merlin.core import analysistask
from merlin.data.codebook import Codebook
from merlin.util import barcodedb, barcodefilters, decoding


class BarcodeSavingParallelAnalysisTask(analysistask.AnalysisTask):
    """An abstract analysis class that saves barcodes into a barcode database."""

    def setup(self, *, parallel: bool, threads: int = 1) -> None:
        super().setup(parallel=parallel, threads=threads)

    def reset_analysis(self, fragmentIndex: int = None) -> None:
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

        self.cropWidth = self.parameters["crop_width"]
        self.imageSize = self.dataSet.get_image_dimensions()

    def get_codebook(self) -> Codebook:
        return self.preprocess_task.get_codebook()

    def run_analysis(self):
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        decode3d = self.parameters["decode_3d"]

        lowPassSigma = self.parameters["lowpass_sigma"]

        codebook = self.get_codebook()
        decoder = decoding.PixelBasedDecoder(codebook)
        scaleFactors = self.optimize_task.load_result("scale_factors")
        backgrounds = self.optimize_task.load_result("background_factors")
        chromaticCorrector = self.optimize_task.get_chromatic_corrector()

        zPositionCount = len(self.dataSet.get_z_positions())
        bitCount = codebook.get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        decodedImages = np.zeros((zPositionCount, *imageShape), dtype=np.int16)
        magnitudeImages = np.zeros((zPositionCount, *imageShape), dtype=np.float32)
        distances = np.zeros((zPositionCount, *imageShape), dtype=np.float32)
        self.barcodes = np.array([], dtype=np.float32).reshape((0, 12 + bitCount))

        if not decode3d:
            for zIndex in range(zPositionCount):
                di, pm, d = self._process_independent_z_slice(
                    self.fragment, zIndex, chromaticCorrector, scaleFactors, backgrounds, self.preprocess_task, decoder
                )

                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = d

        else:
            for zIndex in range(zPositionCount):
                imageSet = self.preprocess_task.get_processed_image_set(zIndex, chromaticCorrector)
                imageSet = imageSet.reshape((imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

                di, pm, _, d = decoder.decode_pixels(
                    imageSet,
                    scaleFactors,
                    backgrounds,
                    lowPassSigma=lowPassSigma,
                    distanceThreshold=self.parameters["distance_threshold"],
                )

                decodedImages[zIndex, :, :] = di
                magnitudeImages[zIndex, :, :] = pm
                distances[zIndex, :, :] = d

            self.barcodes = decoder.extract_all_barcodes(
                decodedImages,
                magnitudeImages,
                None,
                distances,
                self.fragment,
                self.cropWidth,
                zIndex,
                self.global_align_task,
                self.parameters["minimum_area"],
                quick_mode=True
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
        self, fov: int, zIndex: int, chromaticCorrector, scaleFactors, backgrounds, preprocessTask, decoder
    ):
        imageSet = preprocessTask.get_processed_image_set(zIndex, chromaticCorrector)
        imageSet = imageSet.reshape((imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))

        di, pm, npt, d = decoder.decode_pixels(
            imageSet,
            scaleFactors,
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
                    fov,
                    self.cropWidth,
                    zIndex,
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


class SpotDecode(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"warp_task": ["drifts"]})

        self.define_results("fits")

    def norm_image(self, im, s=50):
        im_ = im.astype(np.float32)
        return np.array([im__ - cv2.blur(im__, (s, s)) for im__ in im_], dtype=np.float32)

    def merlin_norm_slice(self, im, s=50):
        fs = int(2 * np.ceil(2 * s) + 1)
        lowpass = cv2.GaussianBlur(im, (fs, fs), s, borderType=cv2.BORDER_REPLICATE)
        im_ = im - lowpass
        im_[lowpass > im] = 0
        return im_

    def merlin_norm_image(self, im, s=50):
        return np.array([self.merlin_norm_slice(im_, s) for im_ in im])

    def get_bit_image(self, fov, channel):
        return np.array(
            [
                self.dataSet.get_raw_image(
                    self.dataSet.get_data_organization().get_data_channel_index(channel),
                    fov,
                    self.dataSet.z_index_to_position(zIndex),
                )
                for zIndex in range(len(self.dataSet.get_z_positions()))
            ]
        )

    def get_local_max(
        self,
        im_dif,
        th_fit,
        im_raw=None,
        dic_psf=None,
        delta=1,
        delta_fit=3,
        dbscan=True,
        return_centers=False,
        mins=None,
        sigmaZ=1,
        sigmaXY=1.5,
    ):
        """Given a 3D image <im_dif> as numpy array, get the local maxima in cube -<delta>_to_<delta> in 3D.
        Optional a dbscan can be used to couple connected pixels with the same local maximum.
        (This is important if saturating the camera values.)
        Returns: Xh - a list of z,x,y and brightness of the local maxima
        """

        z, x, y = np.where(im_dif > th_fit)
        zmax, xmax, ymax = im_dif.shape
        in_im = im_dif[z, x, y]
        keep = np.ones(len(x)) > 0
        for d1 in range(-delta, delta + 1):
            for d2 in range(-delta, delta + 1):
                for d3 in range(-delta, delta + 1):
                    keep &= in_im >= im_dif[(z + d1) % zmax, (x + d2) % xmax, (y + d3) % ymax]
        z, x, y = z[keep], x[keep], y[keep]
        h = in_im[keep]
        Xh = np.array([z, x, y, h]).T
        if dbscan and len(Xh) > 0:
            from scipy import ndimage

            im_keep = np.zeros(im_dif.shape, dtype=bool)
            im_keep[z, x, y] = True
            lbl, nlbl = ndimage.label(im_keep, structure=np.ones([3] * 3))
            l = lbl[z, x, y]  # labels after reconnection
            ul = np.arange(1, nlbl + 1)
            il = np.argsort(l)
            l = l[il]
            z, x, y, h = z[il], x[il], y[il], h[il]
            inds = np.searchsorted(l, ul)
            Xh = np.array([z, x, y, h]).T
            Xh_ = []
            for i_ in range(len(inds)):
                j_ = inds[i_ + 1] if i_ < len(inds) - 1 else len(Xh)
                Xh_.append(np.mean(Xh[inds[i_] : j_], 0))
            Xh = np.array(Xh_)
            z, x, y, h = Xh.T
        im_centers = []
        if delta_fit != 0 and len(Xh) > 0:
            z, x, y, h = Xh.T
            z, x, y = z.astype(int), x.astype(int), y.astype(int)
            im_centers = [[], [], [], [], []]
            Xft = []

            for d1 in range(-delta_fit, delta_fit + 1):
                for d2 in range(-delta_fit, delta_fit + 1):
                    for d3 in range(-delta_fit, delta_fit + 1):
                        if (d1 * d1 + d2 * d2 + d3 * d3) <= (delta_fit * delta_fit):
                            im_centers[0].append((z + d1))
                            im_centers[1].append((x + d2))
                            im_centers[2].append((y + d3))
                            im_centers[3].append(im_dif[(z + d1) % zmax, (x + d2) % xmax, (y + d3) % ymax])
                            if im_raw is not None:
                                im_centers[4].append(im_raw[(z + d1) % zmax, (x + d2) % xmax, (y + d3) % ymax])
                            Xft.append([d1, d2, d3])
            Xft = np.array(Xft)
            im_centers_ = np.array(im_centers)
            bk = np.min(im_centers_[3], axis=0)
            im_centers_[3] -= bk
            a = np.sum(im_centers_[3], axis=0)
            habs = np.zeros_like(bk)
            if im_raw is not None:
                habs = im_raw[z % zmax, x % xmax, y % ymax]

            if dic_psf is not None:
                keys = list(dic_psf.keys())
                ### calculate spacing
                im0 = dic_psf[keys[0]]
                space = np.sort(np.diff(keys, axis=0).ravel())
                space = space[space != 0][0]
                ### convert to reduced space
                zi, xi, yi = (z / space).astype(int), (x / space).astype(int), (y / space).astype(int)

                keys_ = np.array(keys)
                sz_ = list(np.max(keys_ // space, axis=0) + 1)

                ind_ = tuple(Xft.T + np.array(im0.shape)[:, np.newaxis] // 2 - 1)

                im_psf = np.zeros(sz_ + [len(ind_[0])])
                for key in keys_:
                    coord = tuple((key / space).astype(int))
                    im__ = dic_psf[tuple(key)][ind_]
                    im_psf[coord] = (im__ - np.mean(im__)) / np.std(im__)
                im_psf_ = im_psf[zi, xi, yi]
                im_centers__ = im_centers_[3].T.copy()
                im_centers__ = (im_centers__ - np.mean(im_centers__, axis=-1)[:, np.newaxis]) / np.std(
                    im_centers__, axis=-1
                )[:, np.newaxis]
                hn = np.mean(im_centers__ * im_psf_, axis=-1)
            else:
                # im_sm = im_[tuple([slice(x_-sz,x_+sz+1) for x_ in Xc])]
                sz = delta_fit
                Xft = (np.indices([2 * sz + 1] * 3) - sz).reshape([3, -1]).T
                Xft = Xft[np.linalg.norm(Xft, axis=1) <= sz]

                sigma = np.array([sigmaZ, sigmaXY, sigmaXY])[np.newaxis]
                Xft_ = Xft / sigma
                norm_G = np.exp(-np.sum(Xft_ * Xft_, axis=-1) / 2.0)
                norm_G = (norm_G - np.mean(norm_G)) / np.std(norm_G)
                im_centers__ = im_centers_[3].T.copy()
                im_centers__ = (im_centers__ - np.mean(im_centers__, axis=-1)[:, np.newaxis]) / np.std(
                    im_centers__, axis=-1
                )[:, np.newaxis]
                hn = np.mean(im_centers__ * norm_G, axis=-1)

            zc = np.sum(im_centers_[0] * im_centers_[3], axis=0) / np.sum(im_centers_[3], axis=0)
            xc = np.sum(im_centers_[1] * im_centers_[3], axis=0) / np.sum(im_centers_[3], axis=0)
            yc = np.sum(im_centers_[2] * im_centers_[3], axis=0) / np.sum(im_centers_[3], axis=0)
            Xh = np.array([zc, xc, yc, bk, a, habs, hn, h]).T
        if return_centers:
            return Xh, np.array(im_centers)
        return Xh

    def compute_fits(self, image):
        image_norm = self.merlin_norm_image(image, s=3)
        return self.get_local_max(
            image_norm,
            200,
            im_raw=image,
            dic_psf=None,
            delta=1,
            delta_fit=3,
            dbscan=True,
            return_centers=False,
            mins=None,
            sigmaZ=1,
            sigmaXY=1.5,
        )

    def get_inters(self, fits, dinstance_th=2, enforce_color=False):
        """Get an initial intersection of points and save in self.res"""
        res = []
        if enforce_color:
            icols = fits[:, -2]
            for icol in np.unique(icols):
                inds = np.where(icols == icol)[0]
                Xs = fits[inds, :3]
                Ts = cKDTree(Xs)
                res_ = Ts.query_ball_tree(Ts, dinstance_th)
                res += [inds[r] for r in res_]
        else:
            Xs = fits[:, :3]
            Ts = cKDTree(Xs)
            res = Ts.query_ball_tree(Ts, dinstance_th)
        return res

    def get_icodes(self, res, fits, nmin_bits=4, method="top4", redo=False):
        #### unfold res which is a list of list with clusters of loc.

        res = [r for r in res if len(r) >= nmin_bits]
        # rlens = [len(r) for r in res]
        # edges = np.cumsum([0]+rlens)
        res_unfolder = np.array([r_ for r in res for r_ in r])
        # res0 = np.array([r[0] for r in res for r_ in r])
        ires = np.array([ir for ir, r in enumerate(res) for r_ in r])

        ### get scores across bits
        RS = fits[:, -1].astype(int)
        brightness = fits[:, -3]
        colors = fits[:, -2]  # self.XH[:,-1] for bits
        med_cols = {col: np.median(brightness[col == colors]) for col in np.unique(colors)}
        brightness_n = brightness.copy()
        for col in np.unique(colors):
            brightness_n[col == colors] = brightness[col == colors] / med_cols[col]
        scores = brightness_n[res_unfolder]

        bits_unfold = RS[res_unfolder]
        nbits = len(np.unique(RS))
        scores_bits = np.zeros([len(res), nbits])
        arg_scores = np.argsort(scores)

        # print(ires[arg_scores], bits_unfold[arg_scores])
        scores_bits[ires[arg_scores], bits_unfold[arg_scores]] = scores[arg_scores]

        ### There are multiple avenues here:
        #### nearest neighbors - slowest
        #### best dot product - reasonable and can return missing elements - medium speed
        #### find top 4 bits and call that a code - simplest and fastest

        if method == "top4":
            codebook = self.dataSet.load_codebook(0)._data
            codes = np.array([np.where(cd)[0] for cd in codebook.filter(like="bit").to_numpy()])
            codes = [list(np.sort(cd)) for cd in codes]
            gns_names = list(codebook.to_numpy()[1:, 0])
            vals = np.argsort(scores_bits, axis=-1)
            bcodes = np.sort(vals[:, -4:], axis=-1)
            base = [nbits**3, nbits**2, nbits**1, nbits**0]
            bcodes_b = np.sum(bcodes * base, axis=1)
            codes_b = np.sum(np.sort(codes, axis=-1) * base, axis=1)
            icodesN = np.zeros(len(bcodes_b), dtype=int) - 1
            for icd, cd in enumerate(codes_b):
                icodesN[bcodes_b == cd] = icd
            bad = np.sum(scores_bits > 0, axis=-1) < 0
            icodesN[bad] = -1
            igood = np.where(icodesN > -1)[0]
            inds_spotsN = np.zeros([len(res), nbits], dtype=int) - 1
            inds_spotsN[ires[arg_scores], bits_unfold[arg_scores]] = res_unfolder[arg_scores]
            res_prunedN = np.array([inds_spotsN[imol][codes[icd]] for imol, icd in enumerate(icodesN) if icd > -1])
            scores_prunedN = np.array([scores_bits[imol][codes[icd]] for imol, icd in enumerate(icodesN) if icd > -1])
            icodesN = icodesN[igood]
        elif method == "dot":
            icodesN = np.argmax(np.dot(scores_bits[:], self.codes_01.T), axis=-1)
            inds_spotsN = np.zeros([len(res), nbits], dtype=int) - 1
            inds_spotsN[ires[arg_scores], bits_unfold[arg_scores]] = res_unfolder[arg_scores]
            res_prunedN = np.array([inds_spotsN[imol][codes[icd]] for imol, icd in enumerate(icodesN) if icd > -1])
            scores_prunedN = np.array([scores_bits[imol][codes[icd]] for imol, icd in enumerate(icodesN) if icd > -1])

        mean_scores = np.mean(scores_prunedN, axis=-1)
        ordered_mols = np.argsort(mean_scores)[::-1]
        keep_mols = []
        visited = np.zeros(len(fits))
        for imol in ordered_mols:
            r = np.array(res_prunedN[imol])
            r_ = r[r >= 0]
            if np.all(visited[r_] == 0):
                keep_mols.append(imol)
                visited[r_] = 1
        keep_mols = np.array(keep_mols)
        self.scores_prunedN = scores_prunedN[keep_mols]
        self.res_prunedN = res_prunedN[keep_mols]
        self.icodesN = icodesN[keep_mols]

        XH_pruned = fits[self.res_prunedN]
        print(len(self.icodesN))
        return (XH_pruned, self.icodesN, gns_names)
        # XH_pruned -> 10000000 X 4 X 10 [z,x,y,bk...,corpsf,h,col,bit]
        # icodesN -> 10000000 index of the decoded molecules in gns_names
        # gns_names

    def run_analysis(self):
        result = []
        for bit in self.dataSet.get_codebook().get_bit_names():
            image = self.get_bit_image(self.fragment, bit)
            fits = self.compute_fits(image)
            colors = self.dataSet.get_data_organization().data.color.unique()
            bit_index = self.dataSet.get_data_organization().get_data_channel_index(bit)
            color = self.dataSet.get_data_organization().get_data_channel_color(bit_index)
            color_index = np.where(colors == color)[0][0]
            fits = np.concatenate([fits, np.array([[color_index, bit_index]] * len(fits))], axis=-1)
            result.append(fits)
        fits = np.concatenate(result)
        self.fits = fits
        for bit in np.unique(fits[:, -1]):
            drifts = self.warp_task.get_transformation(int(bit))
            if len(drifts) == 3:
                fits[fits[:, -1] == bit, :3] -= drifts
            else:
                fits[fits[:, -1] == bit, 1:3] -= drifts
        inters = self.get_inters(fits)
        XH_pruned, icodesN, gns_names = self.get_icodes(inters, fits)
        savePath = self.dataSet._analysis_result_save_path(
            "decoded", self.analysis_name, self.fragment, subdirectory="decoded"
        )
        np.savez_compressed(savePath, XH_pruned=XH_pruned, icodesN=icodesN, gns_names=np.array(gns_names))
