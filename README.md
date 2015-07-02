# BrainSimInternal
Where Brain Simulator internal development happens.

## VS Solution structure

The solution consists of four major projects:

* **BasicNodes** – A place where you put your C# code and implement a wrapper class for your model
* **BasicNodesCuda** – A place to store your CUDA kernels which are needed for your model execution
* **BrainSimulator** – Simulation front-end, you will alter only configuration of here (hopefully, we can get rid this in future as well)
* **Core** – Core project, you need not to modify it at all
* **MNIST** - Module with MNIST world
* **XmlFeedForwardNet** - Module with feed-forward nets
* **XmlFeedForwardNetCuda** - CUDA kernels for feed-forward nets

