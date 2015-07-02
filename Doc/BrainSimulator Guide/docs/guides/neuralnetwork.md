

# Neural Network

[Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network) are a family of computer learning models, based on biological neural networks (in particular the brain). While learning by example, Neural Networks are capable of estimating any function, even extremely complex ones.

Usually a Neural Network consists of several layers connected in succession. There are different types of layers which each have unique properties. Depending on the task at hand, it can be beneficial to use a certain type or a mix of different layer. In Brain Simulator there is a broad variety of layers available to solve the desired task.

The work flow of a Neural Network is bidirectional. Prediction is usually flowing forward, while learning is flowing backward. In Brain Simulator this meticulous task planning is handled by a group planner like `NeuralNetworkGroup` or `RBMGroup`, which encapsulates the layers.<br>
![](img_examples/NeuralNetworkGroup.PNG)<br>
Check out the [examples](../examples/index.html#neural-network-examples) for different sample implementations of Neural Networks.
