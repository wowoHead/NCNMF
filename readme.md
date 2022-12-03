# NCNMF
This is our implementation for the NCNMF
## 1.Introduce
We proposed a community detection method based on node centrality under the framework of NMF. Specifically, we designed a new similarity measure which considers the proximity of higher-order neighbors to form a more informative graph regularization mechanism, so as to better refine the detected communities. Besides, we introduced the node centrality and \textit{Gini} impurity to measure the importance of nodes and sparseness of the community memberships, respectively. Then, we proposed a novel sparse regularization mechanism which forces nodes with higher node centrality to have smaller \textit{Gini} impurity.
![](https://github.com/wowoHead/NCNMF/tree/master/DATASET/flowchart.png)

## 2.Requirement  
```
Python==3.7
Ubuntu server with 3.70-GHz i9-10900K CPU and 128-GB main memory
```

## 3.Data
|Statistic|Texas|Cornell|Washington|Wisconsin|Gene|Citeseer|Reality-call|BZR|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|#Nodes|185|195|217|262|1,103|3,264|6,809|14,479|
|#Edges|296|286|404|476|1,672|4,612|7,697|15,535|
|#Communities|5|5|5|5|5|2|6|2|10|
