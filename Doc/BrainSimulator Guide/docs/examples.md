# Examples




## Image Processing

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Vision) to peform [image processing](guides/improc.md)

### Segmentation
[Vision/Vision_SLICsegmentation.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_SLICsegmentation.brain)

The sample shows how to segment input image using SLIC algorithm.

### Simple Image Pre-Processing
[Vision/Vision_segmentationOfPhongGame.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_segmentationOfPhongGame.brain)


The sample project shows the full pipeline. First, an input image is segmented into super-pixels (SP). Second, each SP is connected with its neighbors and close-by SP are assigned into a same object id. Third, the attention energy (Ea) is estimated for each object. Fourth, features are estimated as raw image pathces. Fifth, the object features are clustered into a Visual Words to constitute a Working Memory.



## Matrix Node

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Matrix) to peform [matrix oeprations](guides/matrix.md)

### Addition

[Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)

Example with addion of two matrices, the sample also shows row/column wise addion when matrix and vector is added.

### Multiplication

[Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)

Examples with multiplieg two matrix, matrix and vectors. Matrix and constant.

### Log, Exp, Round

[Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_LogExpRound.brain)

Examples with Round, Exp, Log operations on the matrix.

### Get Row / Column

[Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_getRowCol.brain)



## Discrete Q-Learning

### Simple Q-Learning Example
[Q-Learning one reward](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-gridworld-oneReward.brain)

### Composition of two Q-Learning Strategies
[Q-Learning two rewards](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-gridworld-twoRewards.brain)

### Q-Learning plays TicTacToe

[Q-Learning example](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-tictactoe.brain)

## HARM Node examples

[HARM Node example](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/HARM-gridworld-mapG.brain)
