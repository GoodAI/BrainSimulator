# Welcome to BSDoc

The Brain Simulator project (abbr. as BS) is a framework/library/application for effective development/testing of our neural models. It will allow you to:

* Implement your specific model/algorithm as a node that can be used in a larger architecture
* Simulate its behaviour in various environments (called worlds, which provide data sources and outputs/actuators)
* Combine it with algorithms and tools created by others
* Tune your algorithm parameters effectively
* Visualize your data (inputs, outputs, inner memory) with various rendering options
* Implement your own visualizers (called observers) and worlds if needed
* Create projects with complex architectures, save, load and export their states

## Content of this documentation:

* [UI](ui.md) - UI description
* [Model](model.md) - how to implement your own model
* [Observer](observer.md) - how to implement your own observer
* [Persistence](persistence.md) - how persistence works in BrainSimulator
* [Changelog](changelog.md) - changelog and list of future ideas

## Solution architecture

The solution consists of four major projects:

* **TODO: rename to platform? BrainSimulator** – Core project, you need not to modify it at all
* **BrainSimulatorGUI** – Simulation front-end, you will alter only configuration of here (hopefully, we can get rid this in future as well)
* **CUDAKernels** – A place to store your CUDA kernels which are needed for your model execution
* **CustomModels** – A place where you put your C# code and implement a wrapper class for your model

Surely, there will be a situation where existing API won’t be enough for your brand new model and the core project needs to be modified. Please inform me if such situation arise and we will implement it together.



