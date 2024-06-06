Many-Graph Datasets
================

These datsets are composed of many small graphs.
Each is presented as a `torch_geometric.data.InMemoryDataset` object.

Tasks, node features and edge features vary between datasets.
Currently we don't present dynamic graphs, multi-graphs or temporal graphs.

Additionally the functions `get_X_datasets()` retrieve multiple datasets at once.

.. automodule:: datasets
    :members:
    :special-members: __init__
    :show-inheritance:
    :no-inherited-members:
