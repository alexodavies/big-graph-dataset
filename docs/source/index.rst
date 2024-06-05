.. big-graph-dataset documentation master file, created by
   sphinx-quickstart on Tue Jun  4 13:53:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Big Graph Dataset's documentation!
===============================================

This is a collaboration project to build a large, multi-domain set of graph datasets.
Each dataset comprises many small graphs.

.. image:: https://github.com/neutralpronoun/big-graph-dataset/blob/main/outputs/embedding.png
   :alt: embedding image

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
 - Datasets should be many small (<400 node) graphs
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
   :caption: Contents:

   reddit-dataset-example
   datasets
   top

   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
