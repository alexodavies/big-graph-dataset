.. big-graph-dataset documentation master file, created by
   sphinx-quickstart on Tue Jun  4 13:53:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Big Graph Dataset's documentation!
===============================================

This is a collaboration project to build a large, multi-domain set of graph datasets.
Each dataset comprises many small graphs.

The aim of this project is to provide a large set of graph datasets for use in machine learning research.
Currently graph datasets are distributed in individual repositories, increasing workload as researchers have to search for relevant resources.
Once these datasets are found, there is additional labour in formatting the data for use in deep learning.

We aim to provide datasets that are:
 - Composed of many small graphs
 - Diverse in domain
 - Diverse in tasks
 - Well-documented
 - Formatted uniformly across datasets for Pytorch Geometric

Contributing
============

The GitHub repo can be found at `https://github.com/neutralpronoun/big-graph-dataset`.

The basics:
 - Create your own git branch
 - Copy the `datasets/example_dataset.py`
 - Have a look through
 - Re-tool it for your own dataset

I've provided code for sub-sampling graphs and producing statistics.

A few rules, demonstrated in `datasets/example_dataset.py`:
 - The datasets need at least a train/val/test split
 - Datasets should be many small (less than 400 node) graphs
 - Ideally the number of graphs in each dataset should be controllable
 - Data should be downloaded in-code to keep the repo small. If this isn't possible let me know.
 - Please cite your sources for data in documentation - see the existing datasets for example documentation
 - Where possible start from existing datasets that have been used in-literature, or if using generators, use generators that are well-understood (for example Erdos-Renyi graphs)


Getting Started
===============

Check out the Reddit dataset example notebook for a quick start guide, then have a look at the source code for the datasets.

My environment is under `docs/requirements.txt`, use `pip install -r requirements. txt` within a virtual (Conda etc.) environment to get everything installed.

.. toctree::
   :maxdepth: 2

   reddit-dataset-example

Datasets
========

Documentation for the datsets currently in the Big Graph Dataset project.

.. toctree::
   :maxdepth: 2

   datasets

ToP (Topology Only Pre-Training)
================================

Documentation for the Topology Only Pre-Training component of the project.
We are using a pre-trained model to generate embeddings of the graphs in the datasets, hopefully to get some measure of how diverse the datasets are.
Very much a work-in-progress!

.. toctree::
   :maxdepth: 2

   top

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
