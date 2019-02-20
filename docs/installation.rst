Installation
**************

Specifying paths with a .env file
==================================

A .env file is required to specify the search locations for the various input and output files. The following variables should be defined in a file named .env in the project root directory:

* DATA\_HOME - The path of the root directory to the raw data.
* ANALYSIS\_HOME - The path of the root directory where analysis results should be stored.
* PARAMETERS\_HOME - The path to the directory where the merfish-parameters directory resides.

The contents of an example .env file are below:

.. code-block:: none

    DATA_HOME=D:/data
    ANALYSIS_HOME=D:/analysis
    PARAMETERS_HOME=D:/merfish-parameters

Installing prerequisites
==========================

MERlin requires python 3.6 and above. Storm-analysis_ must be installed prior to installing this package. Additionally, the package rtree is not properly installed by pip and should be installed independently. For example, using Anaconda:

.. _Storm-analysis: https://github.com/ZhuangLab/storm-analysis

.. code-block:: none

    conda install rtree

Installing MERlin
==================

MERlin can be installed with pip:

.. code-block:: none

    pip install --process-dependency-links -e MERlin