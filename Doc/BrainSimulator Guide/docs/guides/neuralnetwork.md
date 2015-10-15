

# Neural Network

[Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network) are a family of computer learning models, based on biological neural networks (in particular the structures found in the brain). While learning by example, Neural Networks are capable of estimating any function, even extremely complex ones.

Typically, a Neural Network consists of several layers connected in succession. There are different types of layers which each have unique properties. Depending on the task at hand, one should determine the optimal network structure by changing the layer types, their count or their various parameters.

The work flow of a Neural Network is bidirectional. Prediction is usually flowing forward, while learning is flowing backward. In Brain Simulator, this meticulous task planning is handled by a group planners like `NeuralNetworkGroup` or `RBMGroup`, which encapsulate the layers.

![](img_examples/NeuralNetworkGroup.PNG)

IMPORTANT: *Each network of layers need to be placed in immediate succession and placed inside an appropriate group eg.* `NeuralNetworkGroup`*. This is to ensure that the forward/backward flow planning is executed correctly. If a layer is placed outside the group or if other nodes are placed in between layers, the planning can fail with unexpected results following. At the moment there is no automatic validation of this, so please take care to place layers in succession inside the appropriate group.*

The following Nodes (layers) should be put inside `NeuralNetworkGroup`:

- **Hidden Layer** - most commonly used layer within neural networks. It takes an input and feeds another layer.
- **Output Layer** - output layer takes a target as input, and automatically scales it's neurons to fit the target.
- **Partial output Layer** - output layer as above but only part of the output will be used for update/delta.
- **Q Learning output Layer** - Q learning inside NN.  
- **LSTM Layer** - fully recurrent Long Short Term Memory (LSTM) [1] hidden layer with forget gates and peephole. connections trained by truncated Real-Time Recurrent Learning (RTRL) algorithm.
- **Stack Layer** - Joins two inputs to a single output. Acts as join but allows propagation of deltas. 
- **Gaussian Layer** - Hidden layer where each pair of neurons is interpreted as parameters of Gaussian distribution.
- **Convolutional Layer** - ...
- **Pooling Layer** -

Layers inside `RBMGroup` (Node group used for Restricted Boltzmann Machines and deep learning. Derived from `NeuralNetworkGroup` whose functionality it inherits):

- **RBM input Layer** - Input layer of RBM network. Can only act as a visible layer. 
- **RBM Layer** - One layer of RBM network. Inherited from classical neural hidden layer. Can act as both visible and hidden layer.

Check out the [examples](../examples/neuralnetwork.md) for a variety of implementations of Neural Networks.


## Recurrent Architectures

LSTM [1] are one of the units that supports the reccurent connections in the network. So far, there are two methods that can be used for trainig:

- Real-Time Recurrent Learning (RTRL) [2]
- Back-Propagation Through Time (BPTT) [3]
- RTRL is default and it takes into account previous cell-state of the LSTM, while BPTT is method that remambers all states and wht it updates, it unfold the network in time.

When one wants to use BPTT, the method is following:

- Set the LSTM to perform BPTT.

![](img_examples/BPTT_timeSteps.PNG)

- Set the Network number of time steps to unfold. It will do update in the last time step.

![](img_examples/BPTT_netSettings.PNG)

- When you use observer, it has `Temporal` parameter `TimeStep`. -1 will show the actual value, other will show the corresponidng values for the the desired time-step.

![](img_examples/BPTT_observer.PNG)


## References
[[1] LSTM: A Search Space Odyssey](http://arxiv.org/pdf/1503.04069.pdf)

[[2] Learning Precise Timing with LSTM Recurrent Networks](http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf)

[[3] Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures](http://www.cs.toronto.edu/~graves/nn_2005.pdf)

