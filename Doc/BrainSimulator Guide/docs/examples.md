# Examples




## Image Processing

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Vision) to peform [image processing](guides/improc.md)

### Segmentation
Brain: [Vision/Vision_SLICsegmentation.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_SLICsegmentation.brain)

The sample shows how to segment input image using SLIC algorithm.

![](img/vision_ex_SLIC.PNG)

---

### Simple Image Pre-Processing

Brain: [Vision/Vision_segmentationOfPhongGame.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_segmentationOfPhongGame.brain)

The sample project shows the full pipeline. First, an input image is segmented into super-pixels (SP). Second, each SP is connected with its neighbors and close-by SP are assigned into a same object id. Third, the attention energy (Ea) is estimated for each object. Fourth, features are estimated as raw image pathces. Fifth, the object features are clustered into a Visual Words to constitute a Working Memory.

![](img/vision_ex_pong.PNG)


## Matrix Node

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Matrix) to peform [matrix oeprations](guides/matrix.md).


### Addition

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)

Example with addion of two matrices, the sample also shows row/column wise addion when matrix and vector is added.

![](img/matrix_ex_add.PNG)

---

### Multiplication

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)

Examples with multiplieg two matrix, matrix and vectors. Matrix and constant.

![](img/matrix_ex_multipl.PNG)

---

### Log, Exp, Round

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_LogExpRound.brain)

Examples with Round, Exp, Log operations on the matrix.

![](img/matrix_ex_AbsExp.PNG)

---

### Get Row / Column

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_getRowCol.brain)

![](img/matrix_ex_getRowCol.PNG)



## Discrete Q-Learning

### <a name="qlearningSimple"> Simple Q-Learning Example </a>
[Q-Learning one reward](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-gridworld-oneReward.brain)

### <a name="qlearningTwoNodes"> Composition of two Q-Learning Strategies </a>

[Q-Learning two rewards](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-gridworld-twoRewards.brain)

### <a name="qlearningTicTacToe"> Q-Learning plays TicTacToe </a>

[Q-Learning example](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-tictactoe.brain)

### <a name="harmMapE"> HARM Node example </a>

[HARM Node example](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/HARM-gridworld-mapG.brain)

## Neural Network examples
[XOR gate](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Xor.brain) can be emulated by a feed forward Neural Network.<br>

[MNIST database](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Mnist.brain) is a common testbed for Neural Networks.<br>

[Breakout](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Breakout.brain) based on Q-learning, a type of reinforcement learning.
