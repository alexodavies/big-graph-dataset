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

--- 
| Name  |  Stage  |  Num  |  X shape  |  E shape  |  Y shape  |  Num. Nodes  |  Num. Edges  |  Diameter  |  Clustering  |
|---|---|---|---|---|---|---|---|---|---| 
| ogbg-molpcba | Train | 25000 | 1 | 1 | none | 25.6 ± 6.34| 27.6 ± 7.06| 13.5 ± 3.29| 0.00112 ± 0.011 | 
| facebook_large | Train | 5000 | none | none | 4 | 59.8 ± 20.7| 207.0 ± 169.0| 10.1 ± 6.4| 0.43 ± 0.132 | 
| twitch_egos | Train | 5000 | none | none | 1 | 30.0 ± 11.2| 88.4 ± 71.6| 2.0 ± 0.0| 0.55 ± 0.15 |
| cora | Train | 5000 | 2879 | none | 7 | 59.1 ± 20.6| 118.0 ± 55.2| 10.1 ± 4.66| 0.322 ± 0.0846 | 
| roads | Train | 5000 | 1 | 1 | none | 59.5 ± 20.7| 73.8 ± 27.9| 16.6 ± 6.13| 0.0557 ± 0.0421 |
| fruit_fly | Train | 5000 | none | none | 1 | 59.5 ± 20.6| 207.0 ± 136.0| 6.85 ± 2.24| 0.338 ± 0.0875 | 
| reddit | Train | 1000 | 300 | 86 | 1 | 59.4 ± 20.3| 234.0 ± 176.0| 7.47 ± 4.63| 0.389 ± 0.161 | 
| ogbg-molesol | Val | 1013 | 174 | 13 | 1 | 12.7 ± 6.64| 12.9 ± 7.62| 6.81 ± 3.3| 0.000354 ± 0.00418 | 
| ogbg-molclintox | Val | 1327 | 174 | 13 | 2 | 26.3 ± 15.8| 28.0 ± 17.1| 12.4 ± 6.09| 0.00259 ± 0.0191 | 
| ogbg-molfreesolv | Val | 575 | 174 | 13 | 1 | 8.45 ± 3.96| 8.0 ± 4.48| 5.01 ± 2.08| 0.0 ± 0.0 | 
| ogbg-mollipo | Val | 3778 | 174 | 13 | 1 | 27.0 ± 7.44| 29.4 ± 8.22| 13.8 ± 4.03| 0.00366 ± 0.017 | 
| ogbg-molhiv | Val | 37012 | 174 | 13 | 1 | 25.3 ± 12.0| 27.3 ± 13.1| 12.0 ± 5.15| 0.00158 ± 0.0156 | 
| ogbg-molbbbp | Val | 1833 | 174 | 13 | 1 | 23.5 ± 9.89| 25.4 ± 11.0| 11.2 ± 4.03| 0.00285 ± 0.0278 | 
| ogbg-molbace | Val | 1359 | 174 | 13 | 1 | 34.0 ± 7.88| 36.8 ± 8.12| 15.2 ± 3.14| 0.00664 ± 0.0203 | 
| facebook_large | Val | 1000 | none | none | 4 | 60.2 ± 20.8| 208.0 ± 164.0| 10.2 ± 6.37| 0.433 ± 0.126 | 
| twitch_egos | Val | 5000 | none | none | 1 | 30.0 ± 11.2| 88.4 ± 71.6| 2.0 ± 0.0| 0.55 ± 0.15 | 
| cora | Val | 5000 | 2879 | none | 7 | 59.0 ± 20.7| 118.0 ± 56.5| 10.1 ± 4.77| 0.32 ± 0.0842 | 
| roads | Val | 5000 | 1 | 1 | 1 | 59.6 ± 20.9| 74.1 ± 28.4| 16.4 ± 5.96| 0.0562 ± 0.0422 | 
| fruit_fly | Val | 1000 | none | none | 1 | 59.3 ± 21.0| 213.0 ± 155.0| 6.79 ± 2.21| 0.34 ± 0.0913 | 
| reddit | Val | 1000 | 300 | 86 | 1 | 58.5 ± 20.9| 223.0 ± 179.0| 7.61 ± 4.72| 0.376 ± 0.162 | 
| trees | Val | 5000 | 1 | 1 | 1 | 19.5 ± 6.99| 18.5 ± 6.99| 10.0 ± 3.15| 0.0 ± 0.0 | 
| random | Val | 5000 | 1 | 1 | 1 | 68.9 ± 33.9| 511.0 ± 491.0| 3.89 ± 1.42| 0.17 ± 0.0833 | 
| community | Val | 5000 | 1 | 1 | 1 | 48.0 ± 0.0| 323.0 ± 26.3| 3.0 ± 0.0469| 0.407 ± 0.0251 | 
| ogbg-molesol | Test | 112 | 174 | 13 | 1 | 18.8 ± 6.54| 20.5 ± 7.28| 8.88 ± 3.23| 0.0125 ± 0.0645 | 
| ogbg-molclintox | Test | 147 | 174 | 13 | 2 | 24.6 ± 13.5| 26.7 ± 14.3| 12.0 ± 5.15| 0.00798 ± 0.055 | 
| ogbg-molfreesolv | Test | 64 | 174 | 13 | 1 | 11.1 ± 5.22| 11.8 ± 5.87| 5.27 ± 2.34| 0.0298 ± 0.138 | 
| ogbg-mollipo | Test | 419 | 174 | 13 | 1 | 27.6 ± 7.62| 30.3 ± 8.38| 14.0 ± 4.1| 0.00662 ± 0.0231 | 
| ogbg-molhiv | Test | 4112 | 174 | 13 | 1 | 24.8 ± 12.0| 27.4 ± 13.1| 11.5 ± 5.37| 0.00561 ± 0.0349 | 
| ogbg-molbbbp | Test | 203 | 174 | 13 | 1 | 27.1 ± 13.2| 29.6 ± 14.2| 12.7 ± 5.04| 0.00509 ± 0.0463 | 
| ogbg-molbace | Test | 151 | 174 | 13 | 1 | 34.9 ± 12.8| 37.6 ± 13.1| 15.7 ± 4.99| 0.00509 ± 0.0186 | 
| facebook_large | Test | 1000 | none | none | 4 | 59.0 ± 20.9| 200.0 ± 165.0| 10.2 ± 6.56| 0.425 ± 0.131 | 
| twitch_egos | Test | 1000 | none | none | 1 | 30.1 ± 11.2| 89.7 ± 71.0| 2.0 ± 0.0| 0.554 ± 0.146 | 
| cora | Test | 1000 | 2879 | none | 7 | 60.3 ± 20.8| 122.0 ± 58.3| 10.2 ± 4.53| 0.321 ± 0.084 | 
| roads | Test | 1000 | 1 | 1 | 1 | 59.1 ± 20.7| 73.3 ± 27.7| 16.3 ± 5.95| 0.056 ± 0.0438 | 
| fruit_fly | Test | 1000 | none | none | 1 | 59.9 ± 20.7| 209.0 ± 135.0| 6.82 ± 2.23| 0.338 ± 0.0905 | 
| reddit | Test | 1000 | 300 | 86 | 1 | 59.2 ± 21.3| 235.0 ± 184.0| 7.45 ± 4.67| 0.385 ± 0.16 | 
| trees | Test | 1000 | 1 | 1 | 1 | 19.4 ± 7.14| 18.4 ± 7.14| 9.96 ± 3.19| 0.0 ± 0.0 | 
| random | Test | 1000 | 1 | 1 | 1 | 68.6 ± 34.6| 505.0 ± 487.0| 3.93 ± 1.47| 0.17 ± 0.0839 | 
| community | Test | 1000 | 1 | 1 | 1 | 48.0 ± 0.0| 322.0 ± 25.7| 3.0 ± 0.0316| 0.409 ± 0.0269 | 
