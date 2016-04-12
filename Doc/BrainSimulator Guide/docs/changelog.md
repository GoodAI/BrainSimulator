### Brain Simulator 0.5.0
*Early access release* (2016-04-12)

*Brain Simulator is now licensed under **Apache License**, version 2.0!*

#### New features

**Undo** - it's now possible to undo and redo model changes such as connection or node removal   

**Dynamic model (experimental)** - the ability to programmatically change the model at runtime, add and connect new nodes, etc. (see [dynamic model API description](model.md#dynamic-model))

**Changed memory block dimensions API** - improved API for manipulating with memory block dimensions (the `TensorDimensions` class), removed user defined dimensions from UI (dimensions can be adjusted only on dedicated places such as Join node output or the memory block observer) 

**Brain Unit testing framework (experimental)** - the ability to easily create node and brain tests using Brain Unit node or by defining a simple class (see the [node and brain testing guide](guides\brain_testing.md))

**Memory block metadata** - the ability to attach any metadata to memory blocks (such as preferred visualization method)

**Select backward edges** - select which connections in a cycle are backward to adjust simulation order

**Scripting node group** - a node group that allows to define a custom task scheduler in a script

#### Improvements and fixes

* Options 'Load on start' and 'Save on close' are now saved in the project file
* Fixed and issue which caused the memory block state data to be saved on a wrong place
* Task and task group enabling/disabling can be added to the dashboard
* MNIST World improvements - separate tasks for training and testing data, more improvements
* Improved reduction API

#### Known Bugs

* The CUDA-based Matrix Observer sometimes crashes, please try to replace it with the new CPU-based Matrix Observer (look for an icon with a green dot) 



### Brain Simulator 0.4.0
*Early access release* (2015-12-16)

#### New features

**Dashboard** – a new panel where node and task properties can be pinned and controlled

**Node categories & Toolbar** – node toolbox is now sorted into categories; redesigned node selection UI with improved full-text search

**Profiling** – when turned on, collects profiling information and shows it in the debug/profiling window and the main graph view

**Dynamic memory** – blocks marked as dynamic can be reallocated in node/task code

**Memory blocks dimensions (experimental)** – memory block attribute that allows arbitrary dimension setting both from the UI and from code (please do not use it from code yet, as API will probably change)

##### New worlds and nodes

* **Worlds**
    * Pong for two players – Pong/Breakout game with two paddles and two sets of inputs, playable by two agents
    * Tetris - implementation of the classic Tetris game
    * Genetic Training World / CoSyNE - a world which evolves neural networks to run over continuous environments
* **Nodes**
    * StatisticsNode - mean, variance, and mode (modus); can collect data from previous steps
    * BoxPlotNode - returns five-tuples of min, first quartile, median, third quartile, max; node observer allows visualize output
    * Multiplexer - provides simple data routing; you can set how many steps data should be used for from various inputs, or define simple patterns for such routing

#### Improvements and fixes

* Node status bar (icon strip + topo order) - see topological order and save/load settings at first sight on the node
* Backward edge coloring - edges leading against the topological ordering of nodes are colored red
* CPU observers - rewritten matrix and plot observers so they don't use kernels for drawing
* Disable learning with one click in neural networks - learning of the whole neural network can now be enabled /disabled on the fly with one click
* Debug mode - added breakpoints, step over, step out
* Console autoscroll - the console will auto-scroll when the caret is placed at the end of the text

#### For developers

* CoreRunner (CLI) – allows the creation of custom experiments runnable from command-line without GUI
* MyCudaKernel.RunAsync() - provides the option to call Cuda kernels asynchronously and the possibility to specify another (non-default) Cuda Stream

#### Known Bugs

* Save As and Import does not support moving/loading the state data (e.g. when working with a "brainz" project file)
* Most transformation nodes have incomplete validation that allows zero input; it will cause a crash in kernels during the simulation



### Brain Simulator 0.3.0
*Early access release* (2015-10-29)

#### New Features
 
**AssociativeNetworkWorld**: serves for importing information about some objects / concepts and their relationship to each other into Brain Simulator. In particular, it works with this information in a textual form, such as "the mouse is small." Here, the words "mouse" and "small" represent the concepts and "is" represents the relation between them
 
**CSharpNode and PythonNode**: allows internal scripting inside Brain Simulator. It is possible to program simple tasks right inside the Brain Simulator GUI (without the need to have Visual Studio)
 
**Batch learning**: enables the training of neural networks on multiple data samples (called mini-batches) at one time. Mini-batches help smooth out noise present in individual samples. This can lead to better results and quicker-than-standard online training
 
**Brainz**: makes it possible to save both a project and trained state data into one file

#### More Features and Improvements
 
- **Hiding Observers**: Instant hiding/showing of all observers with a single keyboard shortcut (Ctrl+H). It works even during simulation (without stopping the computation)
- **BPTT** (back-prob through time): The network performs the feed-forward step and "remembers" the output error deltas. Once the sequence is finished, it back-propagates the errors through time and jointly updates weights, taking into account what happens in every time step
- **NN weight sharing**: allows use of the same connection weights in multiple neural networks. This is useful for training a network on one data set and testing it on another at the same time
- **LoopGroup**: node group which runs its content multiple times per simulation step
- **ConditionalGate**: designed for usage together with LoopGroup. It takes two inputs and sends them to its output according to the current loop iteration
 
#### New Worlds
- MyMastermindWorld: a world that simulates the Mastermind game. It outputs both structured data about the game and a visualization of the game board
- AnimationPredictionWorld: a world that loads a sequence of images and presents each simulation step one image from the sequence.
- My2DAgentWorld: simple 2D world where the agent continually moves in 8 directions. The goal is to reach the target, which is placed in a randomly generated position in the space
  
#### Bug Fixes and Updates
- VectorOps: performs 2D vector operations, including rotation and the angle between two vectors
- TextInputNode, TextObserverNode: serves for working with textual information inside Brain Simulator, inputting information manually and visualizing it.
- MyDistanceNode: takes two vectors and outputs a single number as a measure of their similarity. It can compute Euclidean distance, squared Euclidean distance, dot product, cosine distance, Hamming distance/similarity
- MyStackingNode: joins two or more vectors into a single vector.
- Focuser: node has been updated to support a retina-like transformation that uses higher resolution closer to the point of the interest and lower resolution further away
- VariAutoEncoder: Bug fixes have been implemented. VariAutoEncoder is a special case of an Autoencoder model, where a hidden layer is a mixture of gaussians and total loss function is composed of the reconstruction error + regularization error. 
- Distributed reductions or Segmented reductions: Reduction in general is the sum of all values in a vector, and Segmented reduction divides the vector into a set of equally sized vectors and adds each of those vectors separately
- Remove XmlFeedForwardNet: this allowed us to update our Breakout milestone and make it work with new neural net implementations; XMLFFNET is a sub-optimal implementation and we recommend everyone use NNGroup instead
- MyJoin: this node has a new operation called "Equal" which combines two or more vectors of the same length into one output vector. The output vector has a 1 at positions where the two input vectors have the same elements. It has a 0 on the remaining positions
- MatrixNode: Power/exponentiation operation added
- GoniometricFunction: Atan2 operation added
- QLearningLayer: Fixed bug where input and reward of the layer were the same memory block

#### Known Bugs

- Most of **transformation nodes** have incomplete validation that allows zero input. It will cause a crash in kernels during the simulation.
- **Debug mode** limitations
    - Step simulation features do not work
    - Simulation run in debug mode won't load persistent memory blocks data
- **Console view** does not scroll properly. It needs to be focused when scrolling.
- the Save As and Import does not support the moving/loading the state data (e.g. when working with a "brainz" project file)


### Brain Simulator 0.2.0
*Early access release* (2015-08-31)

#### New Features

**TextWorld** - We`ve enabled the use of text as input for Brain Simulator, a feature that was previously unavailable. TextWorld now converts text to a format that neural networks understand, allows the user to choose between two types of sample text inputs, and makes it possible to load external plain text files to be used as input

**SoundWorld** - SoundWorld update is similar to TextWorld, but uses sound rather than text as input. It brings the processing of raw audio samples inside Brain Simulator (for example, for speech recognition tasks)

**StackLayers** - StackLayers provides users with a mechanism for building more complex neural networks. The update enables the stacking of several neural networks in parallel and also putting another layer above them, and connects all networks in the stack to it. This ensures that information between the stack and the higher layer flows correctly

**GaussLayer** - the layer samples from the distribution that is parametrized by the output of the previous layer and this sampling is further used as the input to the following layer. Then it can be applied to sample novel (unseen) data from the distributions.

#### Improvements
- Global data folder is now selected independently from the *Global Load on Start* function

#### Bug Fixes
- Fixed an observer crash when used on an output branch of the fork node
- Prevent node collapse by double click

#### Known Bugs

Same as in the release 0.1. Sorry.-)

### Brain Simulator 0.1
*Early access release* (2015-07-07)

#### Known Bugs

- Most of **transformation nodes** have incomplete validation that allows zero input. It will cause a crash in kernels during the simulation.
- Any kernel crash during simulation makes CUDA context unusable and the application cannot recover from this state yet. Restart of the application is needed. 
- **Debug mode** is not finished
    - Step simulation features do not work
    - Simulation run in debug mode won't load persistent memory blocks data
- The application may crash on attempt to close it during simulation
- **XmlFeedForwardNet** module is obsolete and it may produce unexpected behavior. It is published only for the recent milestone compatibility reasons. Please use **NeuralNetworkGroup** instead.
    - RBM initialization task within the **MyFeedForwardNode** needs to be disabled before the simulation start 
- **Console view** does not scroll properly. It needs to be focused when scrolling.

    