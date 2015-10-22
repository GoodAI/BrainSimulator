## Selected Brain Simulator Worlds

This section shows usage of selected Worlds in the Brain Simulator.

### AnimationPredictionWorld

This world is suitable for testing algorithms that should predict the sequences. The world reads the dataset in in the folowing format: `C:\absolutePath\NamePrefix_00000.png`. Then the world reads the ordered sequence and presents it to the output.

Compared to the `ImageDatasetWorld`, this also supports reloading the dataset online in the `Reload images` Task. This way, it is possible to change the sequence at runtime, without stopping or even pausing the simulation. 

![](../guides/img_examples/animationprediction.gif)


The dataset used in this example can be downloaded [here](../guides/img_examples/SwitchTest.zip).
