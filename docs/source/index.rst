.. big-graph-dataset documentation master file, created by
   sphinx-quickstart on Tue Jun  4 13:53:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:

   self

Big Graph Dataset
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

What we're looking for
=======================

In short: anything! The idea behind this being a collaboration is that we cast a wide net over different domains and tasks.

There are a few rules for this first phase (see below) but the quick brief is that we're looking for datasets of small static graphs with well-defined tasks.
That just means that the structure of the graphs don't vary over time.

If your data is a bit more funky, for example multi-graphs or time-series on graphs, please get in touch and we can discuss how to include it.

In the examples I've provided datasets are mostly sampled from one large graph - this is not compulsory.

Contributing
============

The source can be found in the `Github repository <https://github.com/neutralpronoun/big-graph-dataset>`.

The basics:
 - Create your own git branch
 - Copy the `datasets/example_dataset.py`
 - Have a look through
 - Re-tool it for your own dataset

 See more in :ref:`the Getting Started section<get-started>`.

 .. toctree::
   :maxdepth: 1

   get-started

I've provided code for sub-sampling graphs and producing statistics.

A few rules, demonstrated in `datasets/example_dataset.py`:
 - The datasets need at least a train/val/test split
 - Datasets should be many small (less than 400 node) graphs
 - Ideally the number of graphs in each dataset should be controllable
 - Data should be downloaded in-code to keep the repo small. If this isn't possible let me know.
 - Please cite your sources for data in documentation - see the existing datasets for example documentation
 - Where possible start from existing datasets that have been used in-literature, or if using generators, use generators that are well-understood (for example Erdos-Renyi graphs)

Please document your dataset files with your name and contact information at the top, I'll check code and merge your branches all at once at the end of the project.

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
   :maxdepth: 4

   datasets

ToP (Topology Only Pre-Training)
================================

Documentation for the Topology Only Pre-Training component of the project.
We are using a pre-trained model to generate embeddings of the graphs in the datasets, hopefully to get some measure of how diverse the datasets are.
Very much a work-in-progress!

.. toctree::
   :maxdepth: 2

   top

Credits
=======

This project is maintained by Alex O. Davies, a PhD student at the University of Bristol.
Contributors, by default, will be given fair credit upon initial release of the project.

Should you wish your authorship to be anonymous, or if you have any further questions, please contact me at `<alexander.davies@bristol.ac.uk>`.

.. toctree::
   :maxdepth: 4

   credits

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
