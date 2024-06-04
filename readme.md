# Big Graph Dataset!

This is a collaboration project to build a large, multi-domain set of graph datasets.
Each dataset comprises many small graphs.

![embedding image](https://github.com/neutralpronoun/big-graph-dataset/blob/main/outputs/embedding.png)

## Contributing

The basics:
 - Create your own git branch
 - Copy the `datasets/example_dataset.py`
 - Have a look through
 - Re-tool it for your own dataset

 I've provided code for sub-sampling graphs and producing statistics.

 A few rules, demonstrated in `datasets/example_dataset.py`:
 - The datasets need at least a train/val/test split
 - Datasets should be many small (<200 node) graphs
 - Ideally the number of graphs in each dataset should be controllable
 - Data should be downloaded in-code to keep the repo small. If this isn't possible let me know.
 - Please cite your sources for data in documentation - see the existing datasets for example documentation
 - Where possible start from existing datasets that have been used in-literature, or if using generators, use generators that are well-understood

| Name  |  Stage  |  Num  |  X shape  |  E shape  |  Y shape  |  Num. Nodes  |  Num. Edges  |  Diameter  |  Clustering  | 
|---|---|---|---|---|---|---|---|---|---| 
| ogbg-molpcba | Train | 25000 | 1 | 1 | none | 25.6 ± 6.34| 27.6 ± 7.06| 13.5 ± 3.29| 0.00112 ± 0.011 | 
| facebook_large | Train | 5000 | none | none | 4 | 59.4 ± 20.6| 205.0 ± 171.0| 10.2 ± 6.37| 0.429 ± 0.13 | 
| twitch_egos | Train | 5000 | none | none | 1 | 30.0 ± 11.2| 88.4 ± 71.6| 2.0 ± 0.0| 0.55 ± 0.15 | 
| cora | Train | 5000 | 2879 | none | 7 | 59.6 ± 21.0| 120.0 ± 57.0| 10.1 ± 4.62| 0.323 ± 0.0842 | 
| roads | Train | 5000 | none | none | 1 | 59.4 ± 20.8| 73.7 ± 28.0| 16.4 ± 5.94| 0.0558 ± 0.0414 | 
| fruit_fly | Train | 5000 | none | none | 1 | 59.6 ± 20.9| 208.0 ± 142.0| 6.85 ± 2.31| 0.337 ± 0.0879 | 
| reddit | Train | 1000 | 300 | 86 | 1 | 59.0 ± 20.7| 223.0 ± 178.0| 7.78 ± 4.75| 0.372 ± 0.164 | 
| ogbg-molesol | Val | 1013 | 174 | 13 | 1 | 12.7 ± 6.64| 12.9 ± 7.62| 6.81 ± 3.3| 0.000354 ± 0.00418 | 
| ogbg-molclintox | Val | 1327 | 174 | 13 | 2 | 26.3 ± 15.8| 28.0 ± 17.1| 12.4 ± 6.09| 0.00259 ± 0.0191 | 
| ogbg-molfreesolv | Val | 575 | 174 | 13 | 1 | 8.45 ± 3.96| 8.0 ± 4.48| 5.01 ± 2.08| 0.0 ± 0.0 | 
| ogbg-mollipo | Val | 3778 | 174 | 13 | 1 | 27.0 ± 7.44| 29.4 ± 8.22| 13.8 ± 4.03| 0.00366 ± 0.017 | 
| ogbg-molhiv | Val | 37012 | 174 | 13 | 1 | 25.3 ± 12.0| 27.3 ± 13.1| 12.0 ± 5.15| 0.00158 ± 0.0156 | 
| ogbg-molbbbp | Val | 1833 | 174 | 13 | 1 | 23.5 ± 9.89| 25.4 ± 11.0| 11.2 ± 4.03| 0.00285 ± 0.0278 | 
| ogbg-molbace | Val | 1359 | 174 | 13 | 1 | 34.0 ± 7.88| 36.8 ± 8.12| 15.2 ± 3.14| 0.00664 ± 0.0203 | 
| facebook_large | Val | 1000 | none | none | 4 | 59.3 ± 21.0| 209.0 ± 180.0| 9.59 ± 5.64| 0.429 ± 0.137 | 
| twitch_egos | Val | 1000 | none | none | 1 | 30.1 ± 11.2| 89.7 ± 71.0| 2.0 ± 0.0| 0.554 ± 0.146 | 
| cora | Val | 1000 | 2879 | none | 7 | 60.1 ± 21.0| 121.0 ± 56.7| 10.1 ± 4.39| 0.317 ± 0.0813 | 
| roads | Val | 1000 | none | none | 1 | 60.1 ± 20.6| 74.6 ± 27.6| 16.6 ± 6.18| 0.0545 ± 0.0409 | 
| fruit_fly | Val | 1000 | none | none | 1 | 59.3 ± 20.5| 206.0 ± 135.0| 6.78 ± 2.28| 0.337 ± 0.0873 | 
| reddit | Val | 1000 | 300 | 86 | 1 | 58.5 ± 21.1| 230.0 ± 180.0| 7.42 ± 4.53| 0.389 ± 0.159 | 
| trees | Val | 1000 | 1 | 1 | 1 | 19.4 ± 7.14| 18.4 ± 7.14| 9.96 ± 3.19| 0.0 ± 0.0 | 
| random | Val | 1000 | 1 | 1 | 1 | 68.6 ± 34.7| 504.0 ± 485.0| 3.89 ± 1.39| 0.168 ± 0.083 | 
| community | Val | 1000 | 1 | 1 | 1 | 48.0 ± 0.0| 322.0 ± 25.7| 3.0 ± 0.0316| 0.409 ± 0.0269 | 
| ogbg-molesol | Test | 112 | 174 | 13 | 1 | 18.8 ± 6.54| 20.5 ± 7.28| 8.88 ± 3.23| 0.0125 ± 0.0645 | 
| ogbg-molclintox | Test | 147 | 174 | 13 | 2 | 24.6 ± 13.5| 26.7 ± 14.3| 12.0 ± 5.15| 0.00798 ± 0.055 | 
| ogbg-molfreesolv | Test | 64 | 174 | 13 | 1 | 11.1 ± 5.22| 11.8 ± 5.87| 5.27 ± 2.34| 0.0298 ± 0.138 | 
| ogbg-mollipo | Test | 419 | 174 | 13 | 1 | 27.6 ± 7.62| 30.3 ± 8.38| 14.0 ± 4.1| 0.00662 ± 0.0231 | 
| ogbg-molhiv | Test | 4112 | 174 | 13 | 1 | 24.8 ± 12.0| 27.4 ± 13.1| 11.5 ± 5.37| 0.00561 ± 0.0349 | 
| ogbg-molbbbp | Test | 203 | 174 | 13 | 1 | 27.1 ± 13.2| 29.6 ± 14.2| 12.7 ± 5.04| 0.00509 ± 0.0463 | 
| ogbg-molbace | Test | 151 | 174 | 13 | 1 | 34.9 ± 12.8| 37.6 ± 13.1| 15.7 ± 4.99| 0.00509 ± 0.0186 | 
| facebook_large | Test | 1000 | none | none | 4 | 60.5 ± 20.8| 210.0 ± 166.0| 10.3 ± 6.43| 0.43 ± 0.129 | 
| twitch_egos | Test | 1000 | none | none | 1 | 30.1 ± 11.2| 89.7 ± 71.0| 2.0 ± 0.0| 0.554 ± 0.146 | 
| cora | Test | 1000 | 2879 | none | 7 | 59.9 ± 21.0| 120.0 ± 57.9| 10.3 ± 4.9| 0.322 ± 0.0865 | 
| roads | Test | 1000 | none | none | 1 | 58.7 ± 21.0| 73.0 ± 28.4| 16.2 ± 5.96| 0.0546 ± 0.0411 | 
| fruit_fly | Test | 1000 | none | none | 1 | 59.5 ± 20.7| 203.0 ± 137.0| 6.94 ± 2.4| 0.331 ± 0.0852 | 
| reddit | Test | 1000 | 300 | 86 | 1 | 57.8 ± 20.7| 224.0 ± 179.0| 7.57 ± 4.71| 0.38 ± 0.162 | 
| trees | Test | 1000 | 1 | 1 | 1 | 19.4 ± 7.14| 18.4 ± 7.14| 9.96 ± 3.19| 0.0 ± 0.0 | 
| random | Test | 1000 | 1 | 1 | 1 | 68.5 ± 34.8| 505.0 ± 486.0| 3.86 ± 1.38| 0.17 ± 0.0844 | 
| community | Test | 1000 | 1 | 1 | 1 | 48.0 ± 0.0| 322.0 ± 25.7| 3.0 ± 0.0316| 0.409 ± 0.0269 | 
