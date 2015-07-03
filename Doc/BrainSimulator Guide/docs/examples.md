## Examples




## Image Processing

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Vision) to perform [image processing](guides/improc.md)

### Segmentation
Brain: [Vision/Vision_SLICsegmentation.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_SLICsegmentation.brain)

The sample shows how to segment input image using SLIC algorithm. Note that input is first modified to have a square shape. When one clicks on the observer, there is a `Operation/ObserverMode` property that allows to switch between different visualizations (such as segmentation borders, centers XYZ-color space etc.).

![](img/vision_ex_SLIC.PNG)

---

### Simple Image Pre-Processing

Brain: [Vision/Vision_segmentationOfPhongGame.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_segmentationOfPhongGame.brain)

The sample project shows the full pipeline. First, an input image is segmented into super-pixels (SP). Second, each SP is connected with its neighbors and close-by SP are assigned into a same object id. Third, the attention energy (Ea) is estimated for each object. Fourth, features are estimated as raw image patches. Fifth, the object features are clustered into a Visual Words to constitute a Working Memory.

Again, `Operation/ObserverMode` property of most of observers switches between visualization modes.

![](img/vision_ex_pong.PNG)


## Matrix Node

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Matrix) to peform [matrix oeprations](guides/matrix.md).


### Addition

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)

Brain file with several addition examples (and a comparison of the result to the Join node).
It contains summations of two matrices. Row or column wise addition is included too. Note that dimension of inputs have to correspond.

![](img/matrix_ex_add.PNG)

---

### Multiplication

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_Addition.brain)


Brain file with several multiplication examples.
Again, note that dimension of inputs have to correspond to the desired operation with matrices. The example contains examples with
two memory block inputs as well as with only one, when the second is constant set in `Params\DataInput0`.

![](img/matrix_ex_multipl.PNG)

---

### Log, Exp, Round

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_LogExpRound.brain)

Examples with Round, Exp, Log operations on the matrix. Note that only one input is used in this case. The MatrixNode now applied desired function on that input MemoryBlock.

![](img/matrix_ex_AbsExp.PNG)

---

### Get Row / Column

Brain: [Matrix/Matrix_Addition](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Matrix/Matrix_getRowCol.brain)

This brain file sample shows how to use MatrixNode for getting the desired row or column of the matrix, defined by the id (which row/column I want to get). The observers in the figure bellow shows an example where we want to get row id ,,1'' (so second row because counting starts with 0) of the matrix shown in the middle observer. The result (node's output) is shown in the last observer.

![](img/matrix_ex_getRowCol.PNG)


## Discrete Q-Learning

---
### <a name="qlearningSimple"> Simple Q-Learning Example </a>
Brain:  [QLearning-gridworld-oneReward.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-gridworld-oneReward.brain)

This brain shows basic use of `DiscreteQLearningNode` in the `GridWorld`. The Node receives state description as `GlobalData` (variables + constants) from the World. The reward is defined as a change of the state of the lights ($ values \in \lbrace 0, 1 \rbrace $). The nodes on the left detect changes of variables and select the one for the lights. In the current state, actions are chosen randomly - `GlobalMotivation` is set to 0. Utilities published by the `DiscreteQLearningNode` are multiplied by the `UtilityScaling` value.

![DiscreteQLearningBrain](img/discreteQLearning-brain.PNG)

The following figure shows state of the memory after about 400 simulation steps. It can be seen that the agent **received reward two times**. Also, the agent visited only left part of the World (including door), therefore the $\mathbf{Q}(s,a)$ matrix has currently dimensions only: $10 \times 6 \times 6 ~actions$. The `QLearningObserver` shows:

  * graphical representation of the action with the highest utility.
  * Color corresponds to the value of the utility of the best action (see [Guides section](guides/discreteqlearning.md)).

It can be seen that the Eligibility Trace wrote the $Q$ values on multiple positions back in time. Also we can see that the current strategy already leads towards pressing the switch, despite the fact that it is suboptimal.

![DiscreteQLearning](img/discreteQLearning.PNG)

---
### <a name="qlearningTwoNodes"> Composition of two Q-Learning Strategies </a>

Brain: [QLearning-gridworld-twoRewards.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-gridworld-twoRewards.brain)

The example shows how two different strategies can be composed as described in [Guides section](guides/discreteqlearning.md#harmNode). The task is identical to the brain above, but it has one additional `DiscreteQLearningNode`, which learns different strategy - receives reward when controlling the door. By the `UtilityScaling` sliders it is possible to prioritize between these strategies.

---
### <a name="qlearningTicTacToe"> Q-Learning plays TicTacToe </a>

Brain: [QLearning-tictactoe.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-tictactoe.brain)

The brain can be seen in the figure. The `TicTacToeWorld` only receives actions and sends events indicating whether:

  * the player won
  * the player lost
  * the last player's action was incorrect (position already occupied, take another action)

It has common output indicating state of the game. For each player there is a separate `Event` output and `Action` input. The world sends `Signal` which player should play at a given time step. Each player is placed in the `ConditionalGroup`, which runs the node only if the $signal \in \lbrace PlayerOSignal, PlayerXSignal \rbrace$ is raised.

![TicTacToeBrain](img/ticTacToeBrain.PNG)

Here, the PlayerO `ConditionalGroup` contains `TicTacToePlayerNode` and RL-PlayerX group contains `DiscreteQLearning` Node. The reward signal is defined as follows:

  * try to win
  * avoid losing
  * avoid incorrect moves with lower importance

#### <a name="ticTacToeHowToUse"> How to Use </a>

In this case, the world is "*not passive*". In case that the `RL-PlayerX` produces only random actions it will receive  only punishments most of the time. The following approach works well:

  * Set the `Difficulty` parameter of the `TicTacToePlayerNode` to some lower value (e.g. 0.5)
  * Set the `GlobalMotivation` to some small value (around 0.5).
  * Test the learning convergence:
    * Set the `GlobalMotivation` to 1.0
    * Observe the output of the `Reward+Punishment X` Node in order to see how well the `RL-PlayerX` plays.
    * Around time step 130000, the `RL-PlayerX` should play relatively well against `Difficulty` 0.5 .

---
### <a name="qlearningTicTacToe2"> Two Q-Learning Nodes play TicTacToe</a>
Brain: [QLearning-tictactoe-twoNodes.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/QLearning-tictactoe-twoNodes.brain)

The same task as in the previous example, but in this case, two Nodes learn to play TicTacToe against each other.

---
### <a name="harmMapG"> HARM Node Examples </a>


Brains: [HARM-gridworld-mapF.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/HARM-gridworld-mapF.brain) and [HARM-gridworld-mapG.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/DiscreteQLearning/HARM-gridworld-mapG.brain)

These examples show usage of the `DiscreteHarmNode`, see the [Guides section](guides/discreteqlearning.md#harmNode) for more details.

The `GridWorld` contains an agent, walls and several controlled objects (2 doors and 1 light in this case) and switches which control them. The agent is allowed to use 6 primitive actions $\mathbf{A}=\lbrace Left, Right, Up, Down, Noop, Press \rbrace$. If the agent is on the same position as a switch and executes the $Press$ action, the corresponding switch and its controlled object (e.g. door) change its state. Note: since the `DridWorld` publishes state of each switch and its controlled object separately, the `DiscreteHarmNode` learns two identical strategies for each of these variables.

The following figures illustrate how to use the Node after it has already learned strategies:

  * `ManualOverride` slider is set to 1.0
  * User can choose which `Abstract Actions` should the agent follow by setting their motivation values.


In this case, the motivation is manually set **to turn on the light** - press a button in the left bottom corner.
![HarmSetup](img/harm.PNG)

Note that the `SRPObserver` can be used to show only **two selected dimensions** of a multi-dimensional $\mathbf{Q}(s,a)$ matrix. In this case, the X and Y variables are shown, the rest of dimensions is taken from the current state of the environment. The mark at each position depicts the *action with the maximum utility* value and the mark represents this action.

In the state that the door on the way are closed, the strategy leads first towards the switch that controls these door.

![Lights1](img/harm-lights1.PNG)

After opening the door, the strategy leads directly towards the switch that controls the lights (here, the `SRPObserver` shows different dimension of the matrix).
![Lights2](img/harm-lights2.PNG)


---
## Neural Network examples
[XOR gate](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Xor.brain) can be emulated by a feed forward Neural Network.<br>

[MNIST database](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Mnist.brain) is a common testbed for Neural Networks.<br>

[Breakout](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/Breakout.brain) based on Q-learning, a type of reinforcement learning.

[Recurrent network](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/RNN_sine.brain) predicting sine wave 20 steps ahead.

![](img/RNN_sine.PNG)

[Long Short Term Memory](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/NeuralNetworks/LSTM_sine.brain) predicting sine wave 20 steps ahead.

![](img/LSTM_sine.PNG)

## Motor Control Examples ##

### PID Controller ###
[Arm reaching setpoint with PID controller](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Motor/pid_arm_setpoint.brain). PID controller minimises error between user defined setpoint of arm's joint angles and current arm state.

![](img/pid_arm_setpoint.png)

---

[Balancing inverted pendulum with PID Controller](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Motor/pid_pendulum.brain). PID controller manipulates rotation of the arm so as to keep the pole vertical.

![](img/pid_pendulum.png)

### Jacobian Transpose Control ###


[Bipedal robot doing crunches](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Motor/jtc_bipedal.brain). Setpoint of torso position is moved up and down following a sine wave. PID controller is then used to generate virtual force applied on torso, that would move it to current setpoint. Jacobian transpose control translates virtual force to set of torque for each joint.

![](img/jtc_bipedal.png)
