.. |.genindex| replace:: Index
.. _.genindex: https://big-graph-dataset.readthedocs.io/en/latest/genindex.html
.. |.modindex| replace:: Module Index
.. _.modindex: https://big-graph-dataset.readthedocs.io/en/latest/py-modindex.html
.. |.search| replace:: Search Page
.. _.search: https://big-graph-dataset.readthedocs.io/en/latest/search.html


.. big-graph-dataset documentation master file, created by
   sphinx-quickstart on Tue Jun  4 13:53:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

* `Big Graph Dataset <https://big-graph-dataset.readthedocs.io/en/latest/index.html>`_

  |



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

The source can be found in the `Github repository<https://github.com/alexodavies/big-graph-dataset>`, and documentation on the `readthedocs page<https://big-graph-dataset.readthedocs.io/en/latest/>`.

The basics:
 - Create your own git branch
 - Copy the `bgd/example_dataset.py`
 - Have a look through
 - Re-tool it for your own dataset

 See more in :ref:`the Getting Started section<get-started>`.

 * `Set Up & Contributing <https://big-graph-dataset.readthedocs.io/en/latest/get-started.html>`_



  |



I've provided code for sub-sampling graphs and producing statistics.

A few rules, demonstrated in `bgd/real/example_dataset.py`:
 - The datasets need at least a train/val/test split
 - Datasets should be many small (less than 400 node) graphs
 - Ideally the number of graphs in each dataset should be controllable
 - Data should be downloaded in-code to keep the repo small. If this isn't possible let me know.
 - Please cite your sources for data in documentation - see the existing datasets for example documentation.
 - Where possible start from existing datasets that have been used in-literature, or if using generators, use generators that are well-understood (for example Erdos-Renyi graphs).

Please document your dataset files with your name and contact information at the top, I'll check code and merge your branches all at once at the end of the project.

Getting Started
===============

Check out the Reddit dataset example notebook for a quick start guide, then have a look at the source code for the bgd.

My environment is under `requirements.txt`, use `pip install -r requirements. txt` within a virtual (Conda etc.) environment to get everything installed.
You could also run a conda install using `environment.yml`.

* `Reddit Example Dataset <https://big-graph-dataset.readthedocs.io/en/latest/reddit-dataset-example.html>`_

  * `A walkthrough of the dataset code for the Big Graph Dataset project <https://big-graph-dataset.readthedocs.io/en/latest/reddit-dataset-example.html#A-walkthrough-of-the-dataset-code-for-the-Big-Graph-Dataset-project>`_


  * `Sample to make a dataset of smaller graphs <https://big-graph-dataset.readthedocs.io/en/latest/reddit-dataset-example.html#Sample-to-make-a-dataset-of-smaller-graphs>`_
  * `The final dataset <https://big-graph-dataset.readthedocs.io/en/latest/reddit-dataset-example.html#The-final-dataset>`_
  * `Other datsets <https://big-graph-dataset.readthedocs.io/en/latest/reddit-dataset-example.html#Other-datsets>`_


    |



Datasets
========

Documentation for the datsets currently in the Big Graph Dataset project.

* `Many-Graph Datasets <https://big-graph-dataset.readthedocs.io/en/latest/datasets.html>`_

  * `From Real Data <https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html>`_
  * `Synthetic <https://big-graph-dataset.readthedocs.io/en/latest/datasets/synthetic.html>`_
  * `Functions & Loaders <https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html>`_


    |



ToP (Topology Only Pre-Training)
================================

Documentation for the Topology Only Pre-Training component of the project.
We are using a pre-trained model to generate embeddings of the graphs in the datasets, hopefully to get some measure of how diverse the datasets are.
Very much a work-in-progress!

* `ToP (Topology only Pre-training) <https://big-graph-dataset.readthedocs.io/en/latest/top.html>`_

  |



Credits
=======

This project is maintained by Alex O. Davies, a PhD student at the University of Bristol.
Contributors, by default, will be given fair credit upon initial release of the project.

Should you wish your authorship to be anonymous, or if you have any further questions, please contact me at `<alexander.davies@bristol.ac.uk>`.

* `Credits <https://big-graph-dataset.readthedocs.io/en/latest/credits.html>`_

  |




**Citing**

.. code-block:: bibtex

   @misc{big-graph-dataset,
   title = {{Big Graph Dataset} Documentation},
   howpublished = {https://big-graph-dataset.readthedocs.io/}}


Indices and tables
==================

* |.genindex|_
* |.modindex|_
* |.search|_

