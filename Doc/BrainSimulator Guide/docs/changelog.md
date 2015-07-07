### Current Version

Brain Simulator version **0.1**, *early access release* (7/7/2015).

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

### Recent versions

...
    