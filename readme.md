# Big Graph Dataset!

This is a collaboration project to build a large, multi-domain set of graph datasets.
Each dataset comprises many small graphs.

![embedding image](https://github.com/neutralpronoun/big-graph-dataset/blob/main/outputs/embedding.png)

## Contributing

The basics:
 - Create your own git branch
 - Copy the `example_dataset.py`
 - Re-tool it for your own dataset

 I've provided code for sub-sampling graphs and producing statistics.

 A few rules:
 - The datasets need at least a train/val/test split
 - The training set should have _**no node or edge features**_
 - Datasets should be many small (<200 node) graphs
 - Ideally the number of graphs in each dataset should be controllable

--- 

| Name  |  Stage  |  Num  |  X shape  |  E shape  |  Y shape  |  Num. Nodes  |  Num. Edges  |  Diameter  |  Clustering |
| - | - | - | - | - | - | - | - | - | - |
| ogbg-molpcba | Train | 25000 | 1 | 1 | none | 25.6 ± 6.34| 27.6 ± 7.06| 13.5 ± 3.29| 0.00112 ± 0.011 |
| facebook_large | Train | 5000 | 1 | 1 | none | 59.1 ± 20.6| 200.0 ± 161.0| 10.2 ± 6.4| 0.427 ± 0.131 |
| twitch_egos | Train | 5000 | 1 | 1 | none | 30.0 ± 11.2| 88.4 ± 71.6| 2.0 ± 0.0| 0.55 ± 0.15 |
| cora | Train | 5000 | 1 | 1 | none | 59.7 ± 21.1| 120.0 ± 57.3| 10.2 ± 4.74| 0.322 ± 0.0829 |
| roads | Train | 5000 | 1 | 1 | none | 59.3 ± 20.8| 73.8 ± 28.1| 16.4 ± 6.11| 0.0558 ± 0.0427 |
| fruit_fly | Train | 5000 | 1 | 1 | none | 59.8 ± 20.6| 147.0 ± 103.0| 8.77 ± 3.43| 0.236 ± 0.0868 |
| ogbg-molesol | Val | 1013 | 174 | 13 | 1 | 12.7 ± 6.64| 12.9 ± 7.62| 6.81 ± 3.3| 0.000354 ± 0.00418 |
| ogbg-molclintox | Val | 1327 | 174 | 13 | 2 | 26.3 ± 15.8| 28.0 ± 17.1| 12.4 ± 6.09| 0.00259 ± 0.0191 |
| ogbg-molfreesolv | Val | 575 | 174 | 13 | 1 | 8.45 ± 3.96| 8.0 ± 4.48| 5.01 ± 2.08| 0.0 ± 0.0 |
| ogbg-mollipo | Val | 3778 | 174 | 13 | 1 | 27.0 ± 7.44| 29.4 ± 8.22| 13.8 ± 4.03| 0.00366 ± 0.017 |
| ogbg-molhiv | Val | 37012 | 174 | 13 | 1 | 25.3 ± 12.0| 27.3 ± 13.1| 12.0 ± 5.15| 0.00158 ± 0.0156 |
| ogbg-molbbbp | Val | 1833 | 174 | 13 | 1 | 23.5 ± 9.89| 25.4 ± 11.0| 11.2 ± 4.03| 0.00285 ± 0.0278 |
| ogbg-molbace | Val | 1359 | 174 | 13 | 1 | 34.0 ± 7.88| 36.8 ± 8.12| 15.2 ± 3.14| 0.00664 ± 0.0203 |
| facebook_large | Val | 1000 | 1 | 1 | 1 | 59.9 ± 20.6| 211.0 ± 171.0| 10.1 ± 6.48| 0.436 ± 0.136 |
| twitch_egos | Val | 1000 | 1 | 1 | none | 30.1 ± 11.2| 89.7 ± 71.0| 2.0 ± 0.0| 0.554 ± 0.146 |
| cora | Val | 1000 | 2879 | 1 | 7 | 59.6 ± 20.9| 122.0 ± 58.8| 9.84 ± 4.59| 0.326 ± 0.0863 |
| roads | Val | 1000 | 1 | 1 | 1 | 60.2 ± 21.1| 75.2 ± 29.0| 16.5 ± 5.96| 0.0557 ± 0.0415 |
| fruit_fly | Val | 1000 | 1 | 1 | 1 | 59.8 ± 21.1| 149.0 ± 111.0| 8.87 ± 3.59| 0.235 ± 0.0891 |
| trees | Val | 1000 | 1 | 1 | 1 | 19.4 ± 7.14| 18.4 ± 7.14| 9.96 ± 3.19| 0.0 ± 0.0 |
| random | Val | 1000 | 1 | 1 | 1 | 68.6 ± 34.7| 504.0 ± 484.0| 3.87 ± 1.41| 0.168 ± 0.083 |
| community | Val | 1000 | 1 | 1 | 1 | 48.0 ± 0.0| 322.0 ± 25.7| 3.0 ± 0.0316| 0.409 ± 0.0269 |
| ogbg-molesol | Test | 112 | 174 | 13 | 1 | 18.8 ± 6.54| 20.5 ± 7.28| 8.88 ± 3.23| 0.0125 ± 0.0645 |
| ogbg-molclintox | Test | 147 | 174 | 13 | 2 | 24.6 ± 13.5| 26.7 ± 14.3| 12.0 ± 5.15| 0.00798 ± 0.055 |
| ogbg-molfreesolv | Test | 64 | 174 | 13 | 1 | 11.1 ± 5.22| 11.8 ± 5.87| 5.27 ± 2.34| 0.0298 ± 0.138 |
| ogbg-mollipo | Test | 419 | 174 | 13 | 1 | 27.6 ± 7.62| 30.3 ± 8.38| 14.0 ± 4.1| 0.00662 ± 0.0231 |
| ogbg-molhiv | Test | 4112 | 174 | 13 | 1 | 24.8 ± 12.0| 27.4 ± 13.1| 11.5 ± 5.37| 0.00561 ± 0.0349 |
| ogbg-molbbbp | Test | 203 | 174 | 13 | 1 | 27.1 ± 13.2| 29.6 ± 14.2| 12.7 ± 5.04| 0.00509 ± 0.0463 |
| ogbg-molbace | Test | 151 | 174 | 13 | 1 | 34.9 ± 12.8| 37.6 ± 13.1| 15.7 ± 4.99| 0.00509 ± 0.0186 |
| facebook_large | Test | 1000 | 1 | 1 | 1 | 58.3 ± 20.7| 206.0 ± 178.0| 10.1 ± 6.05| 0.428 ± 0.133 |
| twitch_egos | Test | 1000 | 1 | 1 | none | 30.1 ± 11.2| 89.7 ± 71.0| 2.0 ± 0.0| 0.554 ± 0.146 |
| cora | Test | 1000 | 2879 | 1 | 7 | 60.1 ± 20.7| 120.0 ± 54.6| 10.1 ± 4.71| 0.324 ± 0.0823 |
| roads | Test | 1000 | 1 | 1 | 1 | 59.0 ± 20.8| 73.4 ± 28.1| 16.4 ± 6.06| 0.0556 ± 0.0431 |
| fruit_fly | Test | 1000 | 1 | 1 | 1 | 59.0 ± 20.4| 141.0 ± 88.8| 8.82 ± 3.54| 0.24 ± 0.0848 |
| trees | Test | 1000 | 1 | 1 | 1 | 19.4 ± 7.14| 18.4 ± 7.14| 9.96 ± 3.19| 0.0 ± 0.0 |
| random | Test | 1000 | 1 | 1 | 1 | 68.6 ± 34.7| 505.0 ± 485.0| 3.87 ± 1.36| 0.172 ± 0.085 |
| community | Test | 1000 | 1 | 1 | 1 | 48.0 ± 0.0| 322.0 ± 25.7| 3.0 ± 0.0316| 0.409 ± 0.0269 |