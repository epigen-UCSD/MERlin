Analysis tasks
****************

warp.FiducialFitWarp
---------------------

Description: Aligns image stacks by fitting fiducial spots.

Parameters:

* write\_fiducial\_images -- Flag indicating whether the aligned fiducial images should be saved. These images are helpful for visually verifying the quality of the image alignment.

warp.FiducialCorrelationWarp
-----------------------------

Description: Aligns image stacks by maximizing the cross-correlation between fiducial images. 

Parameters:

* write\_fiducial\_images -- Flag indicating whether the aligned fiducial images should be saved. These images are helpful for visually verifying the quality of the image alignment.

preprocess.DeconvolutionPreprocess
-----------------------------------

Description: High-pass filters and deconvolves the image data in preparation for bit-calling.

Parameters:

* warp\_task -- The name of the warp task that provides the aligned image stacks.
* highpass\_pass -- The standard deviation to use for the high pass filter.
* decon\_sigma -- The standard deviation to use for the Lucy-Richardson deconvolution.

optimize.Optimize
------------------

Description: Determines the optimal per-bit scale factors for barcode decoding.

Parameters:

* iteration\_count -- The number of iterations to perform for the optimization.
* fov\_per\_iteration -- The number of fields of view to decode in each round of optimization.
* estimate\_initial\_scale\_factors\_from\_cdf -- Flag indicating if the initial scale factors should be estimated from the pixel intensity cdf. If false, the initial scale factors are all set to 1. If true, the initial scale factors are based on the 90th percentile of the pixe intensity cdf.

decode.Decode
---------------

Description: Extract barcodes from all field of views.

Parameters:

* crop\_width -- The number of pixels from each edge of the image to exclude from decoding. 
* write_decode_images -- Flag indicating if the decoded and intensity images shold be written.

filterbarcodes.FilterBarcodes
------------------------------

Description: Filters the decoded barcodes based on area and intensity

Parameters:

* area\_threshold -- Barcodes with areas below the specified area\_threshold are removed.
* intensity\_threshold -- Barcodes with intensities below this threshold are removed.  

segment.SegmentCells
----------------------

Description: Determines cell boundaries using a nuclear stain and a cell stain.

Parameters:

* nucleus\_threshold -- The relative intensity threshold for seeding the nuclei.
* cell\_threshold -- The relative intensity threshold for setting boundaries for the watershed.
* nucleus\_index -- The image index for the nucleus image.
* cell\_index -- The image index for the cell image.
* z\_index -- The z index of the nucleus and cell images to use for segmentation.

segment.CleanCellSegmentation
--------------------------------

Description: Cleans the cell segmentation by merging cells that were fargmented along the field of view boundaries.

generatemosaic.GenerateMosaic
-------------------------------

Description: Assembles the images from each field of view into a low resolution mosaic.

Parameters:

* microns\_per\_pixel -- The number of microns to correspond with a pixel in the mosaic.
