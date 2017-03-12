### Genetic Algorithm for Convolutional Neural Networks
CNN architecture exploration using Genetic Algorithm as discussed in the following paper: <a href="https://arxiv.org/abs/1703.01513">Genetic CNN</a>

<p align="center">
<img src="https://github.com/aqibsaeed/Genetic-CNN/blob/master/ga-cnn.png"/>
</p>
<p align="justify">Figure 1: Adapted from Genetic CNN paper. A two-stage network with 4 and 5 nodes at first and second stage respectively. The default input and output nodes are shown in red and green colour respectively. The node connections within light blue region are learned via Genetic Algorithm. The default input node (in red) will be connected to each node without any predecessor in an encoded region. Likewise, the node without successor will be connected to default output node (in green). Also, the nodes without connections will be dropped from the graph (e.g. node B2 in stage 2). Moreover, if a node has more than one parent, they will be summed element-wise before feeding as input to that layer. Within each stage, the number of convolutional filters is a constant and the spatial resolution remains unchanged. The pooling layers are added to down-sample the dataset at each stage.</p>


### Tools Required
Python 3.5 is used during development and following libraries are required to run the code provided in the notebook:

* Tensorflow
* <a href="https://github.com/DEAP/deap">DEAP</a>
* <a href="https://github.com/thieman/py-dag">py-dag</a>

<i>Note: If you see mistakes or want to suggest changes, please submit a pull request.</i>
