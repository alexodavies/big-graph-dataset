.. |CommunityDataset| replace:: ``CommunityDataset``
.. _CommunityDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/synthetic.html#bgd.synthetic.CommunityDataset
.. |compute_top_scores()| replace:: ``compute_top_scores()``
.. _compute_top_scores(): https://big-graph-dataset.readthedocs.io/en/latest/top.html#top.compute_top_scores
.. |CoraDataset| replace:: ``CoraDataset``
.. _CoraDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.CoraDataset
.. |EgoDataset| replace:: ``EgoDataset``
.. _EgoDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.EgoDataset
.. |FacebookDataset| replace:: ``FacebookDataset``
.. _FacebookDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.FacebookDataset
.. |from_ogb_dataset()| replace:: ``from_ogb_dataset()``
.. _from_ogb_dataset(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.from_ogb_dataset
.. |from_tu_dataset()| replace:: ``from_tu_dataset()``
.. _from_tu_dataset(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.from_tu_dataset
.. |FromOGBDataset| replace:: ``FromOGBDataset``
.. _FromOGBDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.FromOGBDataset
.. |FromTUDataset| replace:: ``FromTUDataset``
.. _FromTUDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.FromTUDataset
.. |GeneralEmbeddingEvaluation| replace:: ``GeneralEmbeddingEvaluation``
.. _GeneralEmbeddingEvaluation: https://big-graph-dataset.readthedocs.io/en/latest/top.html#top.GeneralEmbeddingEvaluation
.. |.genindex| replace:: Index
.. _.genindex: https://big-graph-dataset.readthedocs.io/en/latest/genindex.html
.. |get_all_datasets()| replace:: ``get_all_datasets()``
.. _get_all_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_all_datasets
.. |get_datasets()| replace:: ``get_datasets()``
.. _get_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_datasets
.. |get_edge_task_datasets()| replace:: ``get_edge_task_datasets()``
.. _get_edge_task_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_edge_task_datasets
.. |get_graph_classification_datasets()| replace:: ``get_graph_classification_datasets()``
.. _get_graph_classification_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_graph_classification_datasets
.. |get_graph_regression_datasets()| replace:: ``get_graph_regression_datasets()``
.. _get_graph_regression_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_graph_regression_datasets
.. |get_graph_task_datasets()| replace:: ``get_graph_task_datasets()``
.. _get_graph_task_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_graph_task_datasets
.. |get_node_task_datasets()| replace:: ``get_node_task_datasets()``
.. _get_node_task_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_node_task_datasets
.. |get_test_datasets()| replace:: ``get_test_datasets()``
.. _get_test_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_test_datasets
.. |get_train_datasets()| replace:: ``get_train_datasets()``
.. _get_train_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_train_datasets
.. |get_val_datasets()| replace:: ``get_val_datasets()``
.. _get_val_datasets(): https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html#bgd.loaders.get_val_datasets
.. |.modindex| replace:: Module Index
.. _.modindex: https://big-graph-dataset.readthedocs.io/en/latest/py-modindex.html
.. |NeuralDataset| replace:: ``NeuralDataset``
.. _NeuralDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.NeuralDataset
.. |RandomDataset| replace:: ``RandomDataset``
.. _RandomDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/synthetic.html#bgd.synthetic.RandomDataset
.. |RedditDataset| replace:: ``RedditDataset``
.. _RedditDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.RedditDataset
.. |RoadDataset| replace:: ``RoadDataset``
.. _RoadDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/real.html#bgd.real.RoadDataset
.. |.search| replace:: Search Page
.. _.search: https://big-graph-dataset.readthedocs.io/en/latest/search.html
.. |ToPDataset| replace:: ``ToPDataset``
.. _ToPDataset: https://big-graph-dataset.readthedocs.io/en/latest/top.html#top.ToPDataset
.. |TreeDataset| replace:: ``TreeDataset``
.. _TreeDataset: https://big-graph-dataset.readthedocs.io/en/latest/datasets/synthetic.html#bgd.synthetic.TreeDataset


.. big-graph-dataset documentation master file, created by
   sphinx-quickstart on Tue Jun  4 13:53:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

* `Big Graph Dataset <https://big-graph-dataset.readthedocs.io/en/latest/index.html>`_

  |



Big Graph Dataset
===============================================

This is a collaboration project to build a large, multi-domain set of graph bgd.
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

The source can be found in the `Github repository<https://github.com/neutralpronoun/big-graph-dataset>`, and documentation on the `readthedocs page<https://big-graph-dataset.readthedocs.io/en/latest/>`.

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
 - Please cite your sources for data in documentation - see the existing datasets for example documentation
 - Where possible start from existing datasets that have been used in-literature, or if using generators, use generators that are well-understood (for example Erdos-Renyi graphs)

Please document your dataset files with your name and contact information at the top, I'll check code and merge your branches all at once at the end of the project.

Getting Started
===============

Check out the Reddit dataset example notebook for a quick start guide, then have a look at the source code for the bgd.

My environment is under `requirements.txt`, use `pip install -r requirements. txt` within a virtual (Conda etc.) environment to get everything installed.

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

    * |CoraDataset|_


    * |EgoDataset|_


    * |FacebookDataset|_


    * |FromOGBDataset|_


    * |FromTUDataset|_


    * |NeuralDataset|_


    * |RedditDataset|_


    * |RoadDataset|_


    * |from_ogb_dataset()|_
    * |from_tu_dataset()|_

  * `Synthetic <https://big-graph-dataset.readthedocs.io/en/latest/datasets/synthetic.html>`_

    * |CommunityDataset|_


    * |RandomDataset|_


    * |TreeDataset|_



  * `Functions & Loaders <https://big-graph-dataset.readthedocs.io/en/latest/datasets/loaders.html>`_

    * |get_all_datasets()|_
    * |get_datasets()|_
    * |get_edge_task_datasets()|_
    * |get_graph_classification_datasets()|_
    * |get_graph_regression_datasets()|_
    * |get_graph_task_datasets()|_
    * |get_node_task_datasets()|_
    * |get_test_datasets()|_
    * |get_train_datasets()|_
    * |get_val_datasets()|_



      |



ToP (Topology Only Pre-Training)
================================

Documentation for the Topology Only Pre-Training component of the project.
We are using a pre-trained model to generate embeddings of the graphs in the datasets, hopefully to get some measure of how diverse the datasets are.
Very much a work-in-progress!

* `ToP (Topology only Pre-training) <https://big-graph-dataset.readthedocs.io/en/latest/top.html>`_

  * |GeneralEmbeddingEvaluation|_


  * |ToPDataset|_


  * |compute_top_scores()|_


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

