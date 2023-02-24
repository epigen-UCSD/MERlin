# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2019-09-30
### Added
- Iniatialization of this CHANGELOG file to track changes as the version increments

## [0.1.1] - 2019-10-03
### Fixed
- Fixed bug in sum signal

## [0.1.2] - 2019-10-16
### Added
- Exposed tolerance parameter in the adaptive filter barcodes method
- Added plot for scale factor magnitude vs bit index
- Fixed barcode partitioning to include cells from adjacent fields of view when a cell falls across fov boundaries

## [0.1.3] - 2019-12-04
### Fixed
- Addressed bugs present in cleaning overlapping cells and assigning them to a fov
### Added
- Added option to draw field of view labels overlaid on the mosaic

## [0.1.4] - 2019-12-05
### Added
- Added task to evaluate whether a parallel analysis task has completed
### Changed
- Changed the clean overlapping cells to run in parallel
- Snakemake job inputs were simplified using the ParallelCompleteTask to improve DAG construction speed and overall snakemake runtime performance

## [0.1.5] - 2020-01-22
### Changed
- Updated the filemap to only store the file name so that it can easily be pointed to new data home directories. This change maintains backward compatibility.
- Improved decoding speed
### Added
- Parameters to filter tasks that enable removing barcodes that were putatively duplicated across adjacent z planes.

## [0.1.6] -
### Fixed
- Fixed bug and edge cases in removal of barcodes duplicated across z planes. Moved to the decode step to prevent unintended conflict with misidentification rate determination.

### Added
- An alternative Lucy-Richardson deconvolution approach that requires ~10x fewer iterations.

## Changes at UCSD Center for Epigenomics
### Fixed
- Fixed problems with newer versions of certain packages so that MERlin can now run with the current versions of all packages installed.
- Fixed an issue we were having with the 3D decoding option.
- Fixed a problem where the randomness in the Optimize task would cause MERlin to refuse to restart on an existing folder after a crash or stop.
- Removed various warnings printed from deprecated functions, division by zero, etc.
- Fixed some edge case crashes caused by no decoded barcodes in a FOV.
### Added
- Added a FiducialBeadWarp task that performs drift correction more robustly than FiducialCorrelationWarp. Only suitable for registration using fiducial beads.
- Added zarr file format support for raw images.
- Added option to warp.py tasks to set the imaging round that all rounds are aligned to.
- Image files can now be organized into sub-folders (e.g. one per hyb round).
- FOVs can now be identified by arbitrary strings instead of sequential integers.
- FOVs can be identified by different parts of the filename by matching fov1, fov2, etc. in the data organization file.
- AlignDAPI3D task added for doing 3D drift correction based on DAPI images.
