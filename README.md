# Conagraph

Here is an implementation of Conagraph, which implements contrastive
learning on aggregated log graph (ALG), building an ALG based on
action-based logs, implementing contrastive learning through maximiz-
ing mutual information to get the embeddings, and then detecting the
anomalies.

## Datasets

Considering the file size of the datasets, the datasets are not given here. We use 2 public datasets:

1. Kent, A.D.: Cyber security data sources for dynamic network research. In: Dynamic
Networks and Cyber-Security. pp. 37–65 (2016), https://www.worldscientific.com/
doi/abs/10.1142/9781786340757_0002
2. Lindauer, B.: Insider threat test dataset (2020), https://kilthub.cmu.edu/articles/
dataset/Insider_Threat_Test_Dataset/12841247

And a real-world dataset.

## Description

* _framework/alg_generator/_ : generation process of alg.
* _framework/conag/_ : basic model framework of Conagraph.
* _framework/execution/_ : data preparation, training and testing process.
* _framework/graph/_ : node class and edge clas.
* _framework/layers/_ : layers used by the network.
* _global_object/_ : dataset file path and process value file(could be generated) path. Setting required.

## License

MIT
