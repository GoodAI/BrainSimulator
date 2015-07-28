## Nupic Node

This sample project is distributed together with the NupicNode through a separate [NupicModule repository](https://github.com/GoodAI/NupicModule), because of different licensing (GPL v3). The sample project file is located in the `BrainSimulatorExample` folder.

The NupicModule contains the Numenta's [nupic.core](https://github.com/numenta/nupic.core) C++ codes, needed external libraries and a C# wrapper, that allows to run the nupic.core functions from inside of the Brain Simulator node (`MyNupicNode.cs`, `MyNupicTasks.cs`). We made use of the [Numenta's open source project](https://github.com/numenta) in order to have the official Cortical Learning Algorithms (CLA) implementation, so we can have a direct comparison and we can easily utilize the future changes from the Nupic community.

### Installation of the Nupic Node

To start to use the NupicModule you have to:

 * have the Brain Simulator installed
 * clone the NupicModule repository (or download and unpack the zip with sources)
 * create the project file `NupicModule.csproj.user` in the folder `NupicModule\Module` for the NupicModule project; this can be created from the `NupicModule.csproj.user.template-RENAME_ME` file by removing the postfix in its name
 * open the `NupicModule\NupicModule.sln` in MS VisualStudio 2013 and edit the paths in NupicModule project's Properties/Debug tab so they point to your Brain Simulator installation and to your location of the NupicModule
 * now hitting StartDebugging (F5) should build everything, launch the installed Brain Simulator and load the NupicModule.dll into it

### CLA MNIST Prediction Example

Brain: [NupicModule\BrainSimulatorExample\CLA_mnist_prediction.brain](https://github.com/GoodAI/NupicModule/blob/master/BrainSimulatorExample/CLA_mnist_prediction.brain)

If the NupicModule is set up correctly, you can now load the `NupicModule\BrainSimulatorExample\CLA_mnist_prediction.brain` project, run the simulation and you shall see the following workspace:
![](../img/nupicNode.PNG)
The project is set up for the MNIST hand-written numbers recognition and series prediction using the CLA algorithm  implemented in the NupicNode. The available inputs are the standard 28 * 28 MNIST bitmap coded with float numbers (0 to 1) and the Label, which is the integer number that is represented on the bitmap. First, we do some pre-processing with the nodes `Prescaling` and `ProcessedMnistInput` in order to have the bitmap in a correct format for the NupicNode input - it expects a binary (0,1) vector. We connect the processed MNIST Bitmap and the input Label to the NupicNode's inputs Input and Label. There are several **outputs** from the NupicNode:

 * `ActiveColumns` - a vector of active columns (the learned representation of the current input)
 * `TP` - output of the Temporal Pooler part of the algorithm (vector of cells that are in predictive state)
 * `Classifier` - shows the content of the classifier's buckets - each bucket contains probability value representing how sure the classifier is that this bucket will be active in the predicted time; several buckets can have non-zero values
 * `ClassifierBestPredictions` - contains the single label assigned to the most probable bucket

In the example brain file, we are predicting only a single time step that is 1 step in the future (this is set up in the NupicNode's property `PredictedStepsList` - it can contain a list of more comma separated time step values and predict several of them simultaneously). This value - the NupicNode's prediction is visualised in the top left observer. Below it is shown the actual input into the NupicNode. The input is being sent in a sequential way, every digit from 0 to 9 is sent one after another (after 9 it wraps up to 0 again), so the correct prediction should be exactly by 1 higher than the one in the current input.

#### NupicNode Parameters

The NupicNode has plenty of parameters to fine tune, which is very impractical to do manually. The actual parameters used in this example were obtained using the [swarming optimization algorithm](https://github.com/numenta/nupic/wiki/Swarming-Algorithm), present in the Nupic framework (currently working only in Linux/MacOS). The core CLA parameters were exported into the [nupic_params_from_swarm.txt file](https://github.com/GoodAI/NupicModule/blob/master/BrainSimulatorExample/nupic_params_from_swarm.txt) and are automatically loaded and set to the NupicNode through its property `PropertiesFileName`. The description of the swarming process is beyond the scope of this document, but you can find it on [Nupic's Github pages](https://github.com/numenta/nupic/wiki/Running-Swarms).

#### Results

The bottom right observer `CLA_1_StepPrediction - ClassifierOutput` depicts the probability of different buckets. We can see, that sometimes the classifier assigns high probabilities to more buckets, i.e. when the current input is 9, it can predict both, 0 and 8 with a high probability, because they are similar from the NupicNode's point of view. The rightmost node is a NodeGroup that contains just several nodes for the calculation of error. The sliding average of the error over the last 100 time steps is plotted in the largest `TimePlotObserver: AverageErrorOverLast100Steps`. After starting the simulation, we can see that the error peaks somewhere around 100 steps, that means that the NupicNode needs to see about 100 input values (each digit 10 times) until it learns to recognize and predict them. The top right observer prints the current average error value and the one below it prints the total number of misses since the start of the simulation. The average error rate of prediction is around 7% in this experiment.

#### Discussion, Input Encoding

When we look on the setting of the `MNISTWorld`, into the parameter `ImagesCnt` of the task `Init MNIST World`, we can see the total number of input images is 100 (10 for each digit). The algorithm doesn't work very well with larger number. This shows the largest disadvantage of this setting of the experiment. The NupicNode with these parameters (mainly limited by the number of columns) and this format of input is not able to learn and more importantly generalize larger number of different shapes of digits. The main contributor to this is probably the encoding of the input MNIST bitmaps, which is not very appropriate for the CLA algorithm. The CLA requires binary input vectors with **Sparse Distributed Representation** (SDR) properties, e.g. containing much lower number of 1s than 0s (around 2%) and the meaning highly correlated with the position of 1s in the vector. The testing and creation of better encodings will be the next step in our research. Currently there is only one simple encoder present in the Brain Simulator BasicNodes - the `ScalarToSDRNode` which encodes scalar numbers into SDRs, with which you can experiment. Nevertheless, the example shows that the CLA can very quickly learn to accurately recognize and predict the patterns in a time series.

