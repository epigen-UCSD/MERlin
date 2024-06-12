[![DOI](https://zenodo.org/badge/202668055.svg)](https://zenodo.org/badge/latestdoi/202668055)

# MERlin - Extensible pipeline for scalable data analysis

MERlin is an extensible data analysis pipeline for reproducible and scalable analysis of large 
datasets. Each MERlin workflow consists of a set of analysis tasks, each of which can be run as 
single task or split among many subtasks that can be executed in parallel. MERlin is able to 
execute workflows on a single computer, on a high performance cluster, or on the cloud 
(AWS and Google Cloud).

If MERlin is useful for your research, consider citing:
Emanuel, G., Eichhorn, S. W., Zhuang, X. 2020, MERlin - scalable and extensible MERFISH analysis software, v0.1.6, Zenodo, doi:10.5281/zenodo.3758540 

This fork of MERlin contains additions and modifications made at the UCSD Center for Epigenomics. These changes include:
* Additional drift correction methods which can correct drift in the z-axis and support correcting drift based on either DAPI staining or fiducial beads. 
* Integration of Cellpose segmentation.
* FinalOutput task which generates a scanpy object and initial cell clustering.
* General performance improvements and bug fixes.
* Changes to fix compatibility issues with newer versions of various python packages.

## Authors

### Xiaowei Zhuang lab, Harvard (original MERlin authors)
* [**George Emanuel**](mailto:emanuega0@gmail.com) - *Initial work* 
* **Stephen Eichhorn**
* **Leonardo Sepulveda**

### University of California, San Diego
* [**Colin Kern**](mailto:jckern@health.ucsd.edu) (Center for Epigenomics)
* New drift correction methods: Bogdan Bintu (Department of Cellular and Molecular Medicine)
