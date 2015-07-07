## Vector Symbolic Architecture

This sample project shows how high level symbols can be created with high-dimensional real vectors and manipulated through various operators ([binding.brain](https://github.com/GoodAI/BrainSimulatorSampleProjects/tree/master/VSA/binding.brain))

### Introduction

Vector symbolic representation is a very useful model when dealing with cognitive processes related tasks. For further details, please read some literature related to this topic [1,2,3].

There are several basic operations and processes connected to vector symbolic architecture (VSA) and there are nodes for each area in Brain Simulator 
*(there are also other nodes but we will focus on basic ones in this example)*. 

* Symbol **creation** (`CodeBook`)
* Symbol **manipulation** (`BindingNode`)
* Symbol **testing** (`CodeBook`)

The `CodeBook` has two modes. In the default mode, it will **create a symbol** form predefined code book. In the testing mode, it can **evaluate dot product** similarity between selected predefined symbol and the node input.

The `BindingNode` also has two modes.It can **bind** its inputs together or **unbind** (release) them apart. Notice that bind operation is commutative but unbind is not. You are always unbinding with the second input.

Moreover, you can use any other **vector operations** (addition, subtraction, normalization etc.) in order to transform vectors when needed.

### Working memory example

In this example, you can examine simple working memory model. We want to store any piece of information coming from inputs into the working memory, be able to hold it there for a while and forget it when it is not available on input anymore.

There are two sources (senses) for the content of the working memory. You can either *see* or *hear* things. On the left side of the model you can watch how perceived items are constructed and bound with proper context. You can **turn on/off** the item appearance in the memory with setting highlighted function node to `f(x) = 0`).

![Binding](../img/binding.png) 

After memory items are created and bound they can be added together into a superposition and sent to working memory. The following picture shows the structure of the working memory group. 

![Working Memory](../img/working_mem.png)

As you can see, items entering working memory are normalized, some random noise is added to it before storing into the working memory and some more noise is added after retrieving as well. Notice that the magnitude of the noise is equal to magnitude of the input vector. The memory accumulator node represents the working memory. Its only task is to make and approximation of moving average of items going in.

The last part of the models tries to recall items from the working memory and shows the **signal strength** of any detected item through plot observers. Each curve represents the detection level for each predefined memory item. If you play with **turn on/off** controls you can see how items are appearing and disappearing from memory. The plot on the left side shows visual features (what I can see) and the plot on the right side shows audio features. The observer in the middle shows the real values inside the working memory during simulation.

![Working Memory](../img/wm_detection.png)

[1] [Tony Plate, *Holographic Reduced Representations: Convolution Algebra for Compositional Distributed Representations*](http://www.ijcai.org/Past%20Proceedings/IJCAI-91-VOL1/PDF/006.pdf)

[2] [Matthew A. Kelly, Robert L. West, *From Vectors to Symbols to Cognition:
The Symbolic and Sub-Symbolic Aspects of Vector-Symbolic Cognitive Models*](https://mindmodeling.org/cogsci2012/papers/0311/paper0311.pdf)

[3] [Pentti Kanerva, *Fully Distributed Representation*](http://www.rni.org/kanerva/rwc97.ps.gz)
