import argparse
import logging
import numpy as np
import torch
from torch_geometric.transforms import Compose
from datasets import get_train_datasets, get_val_datasets, get_test_datasets, get_all_datasets
from utils import *
from top import compute_top_scores, ToPDataset


def desc_datasets(datasets, stage, dataset_names):
    these_print_strings = []
    for i_dataset, dataset in enumerate(datasets):
        arrays, metrics, names = get_metric_values(dataset)
        print_string = f"{dataset_names[i_dataset]} & {stage} & {len(dataset)}"

        one_sample = dataset[0]

        x_shape, edge_shape, y_shape = one_sample.x, one_sample.edge_attr, one_sample.y
        x_shape = x_shape.shape[1] if x_shape is not None else "none" 
        edge_shape = edge_shape.shape[1] if edge_shape is not None else "none" 
        try:
            y_shape = y_shape.shape if y_shape is not None else "none" 
            if y_shape != "none":
                if len(y_shape) == 2:
                    y_shape = y_shape[1]
                else:
                    y_shape = y_shape[0]
        except:
            y_shape = "none"

        print_string += f" & {x_shape} & {edge_shape} & {y_shape} "
        for i_name, name in enumerate(names):
            value, dev = np.mean(arrays[i_name]), np.std(arrays[i_name])
            value = float('%.3g' % value)
            dev = float('%.3g' % dev)
            print_string += f"& {value} $\pm$ {dev}"
        print(print_string + r"\\")
        these_print_strings.append(print_string)

    print("\n")
    return these_print_strings

def latex_to_markdown(strings):
    new_string = ""
    for itxt, txt in enumerate(strings):
        
        txt = txt.replace("&", "|")
        txt = txt.replace(r"\\", "")
        txt = txt.replace("$", "")
        txt = txt.replace("\pm", "±")

        print("| " + txt + " |")
        new_string += "| " + txt + " | \n"

        if itxt == 0:
            print("|---|---|---|---|---|---|---|---|---|---|")
            new_string += "|---|---|---|---|---|---|---|---|---|---| \n"

    return new_string

def update_readme_table(readme_path, new_table):
    with open(readme_path, 'r') as file:
        readme = file.readlines()

    # Find the start and end of the table
    table_start = None
    table_end = None
    for i, line in enumerate(readme):
        if ' Name  ' in line:
            table_start = i - 1
        elif table_start is not None and line.strip() == '':
            table_end = i
            break

    if table_end is None:
        table_end = len(readme)

    if table_start is None or table_end is None:
        raise ValueError('Could not find table in README')

    # Split the new table into lines and add the newline character back to the end of each line
    new_table_lines = [line + '\n' for line in new_table.split('\n') if line]

    # Replace the table with the new table
    readme = readme[:table_start] + new_table_lines + readme[table_end:]
    
    # Write the updated README back to the file
    with open(readme_path, 'w') as file:
        file.write(''.join(readme))

def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)


    # Get datasets
    my_transforms = Compose([initialize_edge_weight])

    train_datasets, train_names = get_all_datasets(my_transforms, num = args.num_train)
    top_datasets = [ToPDataset(dataset) for i, dataset in enumerate(train_datasets)]
    compute_top_scores(top_datasets, train_names)

    roots = []
    for idataset, dataset in enumerate(train_datasets):
        try:
            roots.append(dataset.root)
        except AttributeError:
            train_datasets[idataset] = dataset.datasets[0]
            roots.append(train_datasets[idataset].root)
            # roots.append(dataset.datasets[0].root)

    for i, dataset in enumerate(train_datasets):
        print(dataset, train_names[i], roots[i])
        vis_grid(dataset[:25], f"{roots[i]}/{train_names[i]}.png")

        

    train_datasets, train_names = get_train_datasets(my_transforms, num = args.num_train)

    val_datasets, val_names = get_val_datasets(my_transforms, num = args.num_val)
    test_datasets, test_names = get_test_datasets(my_transforms, num = args.num_test)
    print_strings = [r"Name  &  Stage  &  Num  &  X shape  &  E shape  &  Y shape  &  Num. Nodes  &  Num. Edges  &  Diameter  &  Clustering \\"]
    print(print_strings[0])

    print_strings += desc_datasets(train_datasets, "Train", train_names)
    print_strings += desc_datasets(val_datasets, "Val", val_names)
    print_strings += desc_datasets(test_datasets, "Test", test_names)

    md_string = latex_to_markdown(print_strings)
    update_readme_table("README.md", md_string)








def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--num_train', type=int, default=5000,
                        help='Number of points included in the train datasets')
    
    parser.add_argument('--num_val', type=int, default=1000,
                    help='Number of points included in the validation datasets')

    parser.add_argument('--num_test', type=int, default=1000,
                    help='Number of points included in the test datasets')
    
    parser.add_argument(
        '-s',
        '--score',
        action='store_true',
        help='Whether to compute similarity score against other datasets',
        default = False
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    run(args)

