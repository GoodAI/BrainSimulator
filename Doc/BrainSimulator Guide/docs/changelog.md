### Brain Simulator 0.2.0
*Early access release* (31/8/2015)

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
*Early access release* (7/7/2015)

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

    