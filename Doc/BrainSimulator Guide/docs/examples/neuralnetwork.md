## Neural Network examples
[XOR gate](https://github.com/GoodAI/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Xor.brain) can be emulated by a feed forward Neural Network. This example shows the actual observed error.<br>
![](../img/XOR_flow.PNG)

[MNIST database](https://github.com/GoodAI/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Mnist.brain) is a common testbed for Neural Networks. This example shows the moving average of the observed error.<br>
![](../img/MNIST_flow.PNG)

[Breakout](https://github.com/GoodAI/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Breakout.brain) based on Q-learning, a type of reinforcement learning. This example includes a gate between learned and random outputs.
![](../img/Breakout_flow.PNG)

[Recurrent network](https://github.com/GoodAI/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/RNN_sine.brain) predicting sine wave 20 steps ahead.

![](../img/RNN_sine.PNG)

[Long Short Term Memory](https://github.com/GoodAI/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/LSTM_sine.brain) predicting sine wave 20 steps ahead.

![](../img/LSTM_sine.PNG)

[Variational AutoEncoder.](https://github.com/GoodAI/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/LSTM_sine.brain) Guassian sampling node is between the encoder and decoder. If the output of the encoder has size $N$, first $N/2$ elements represents the mean ($\mu$) and second $N/2$ the variance ($\sigma$) that is the parametrization of a Gaussian distribution. Sampling node just sample from the distribution and this sampling is further used as the input to the decoder. See [1] for more details. The slider (outside the Group) turns off the training and the Gaussian node randomly generates samples.

![](../img/Vari_AutoEncoder_brain.PNG)



### References
 [1] [Diederik P Kingma, Max Welling. Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114)
