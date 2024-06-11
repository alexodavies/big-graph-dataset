Many-Graph Datasets
===================


.. image:: _static/datasets.png
   :alt: Examples of the currently included datasets
   :width: 800px


These datsets are composed of many small graphs.
Each is presented as a `torch_geometric.data.InMemoryDataset` object.

Tasks, node features and edge features vary between datasets.
Currently we don't present dynamic graphs, multi-graphs or temporal graphs.

Additionally the functions `get_X_datasets()` retrieve multiple datasets at once.

.. toctree:: 
   :maxdepth: 2

   datasets/real
   datasets/synthetic
   datasets/loaders


**Citing**

.. code-block:: bibtex

   @misc{big-graph-dataset,
   title = {{Big Graph Dataset} Documentation},
   howpublished = {https://big-graph-dataset.readthedocs.io/}}