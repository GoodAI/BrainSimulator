# Restricted Boltzmann Machine

Restricted Boltzmann Machine (**RBM**) is a stochastic autoencoder that can serve as feature encoder and/or decoder.

One of its uses is initialization of weights of a neural network prior to its actual training via stochastic gradient descent (SGD), using e.g. back-propagation for (fine) tuning.

When used like this, RBMs are stacked on top of each other to form a deep belief network (DBN).

This is why RBM in Brain Simulator is built upon the classic neural network hidden layer. RBM layer inherits its whole functionality and thus can be used as both autoencoder and as a part of any neural network.


## Nodes

### RBM Group

Any Brain Simulator network that uses RBM layers must be created under the `RBMGroup` node.

The group itself handles proper flow of data inside the network as long as it is in the RBM mode.  
The other possibility is to use various types of SGD available in the group, since the group itself is a descendant of the Neural Network group. Both modes can be swapped during runtime.

#### Parameters

### RBM layers



#### Parameters


## Algorithms

### Training

### Reconstruction

## Example

There is an example tutorial as well as a brain file with a two-layer RBM trained on the MNIST dataset.
