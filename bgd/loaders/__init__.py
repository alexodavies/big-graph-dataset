from .loaders import get_test_datasets, get_val_datasets, get_train_datasets, get_all_datasets
from .loaders import get_graph_task_datasets, get_edge_task_datasets, get_node_task_datasets
from .loaders import get_graph_classification_datasets, get_graph_regression_datasets, get_datasets


__all__ = ['get_test_datasets', 'get_val_datasets', 'get_train_datasets', 'get_datasets',
            'get_all_datasets', 'get_graph_task_datasets', 'get_edge_task_datasets',
              'get_node_task_datasets', 'get_graph_classification_datasets', 'get_graph_regression_datasets']