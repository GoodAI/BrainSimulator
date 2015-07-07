## Image Processing

These sample projects show several [examples](https://github.com/GoodAI/BrainSimulatorSampleProjects/tree/master/Vision) to perform [image processing](../guides/improc.md)

### Segmentation
Brain: [Vision/Vision_SLICsegmentation.brain](https://github.com/GoodAI/BrainSimulatorSampleProjects/tree/master/Vision/Vision_SLICsegmentation.brain)

The sample shows how to segment input image using the SLIC algorithm. Note that the input image is first re-sized to have a square shape, which is a requirement of the segmentation node. When one clicks on the observer, there is a `Operation/ObserverMode` property that yields to switch between different visualizations modes (such as segmentation borders, centers XYZ-color space etc.).

![](../img/vision_ex_SLIC.PNG)

---

### Simple Image Pre-Processing

Brain: [Vision/Vision_segmentationForBreakout.brain](https://github.com/GoodAI/BrainSimulatorSampleProjects/tree/master/Vision/Vision_segmentationForBreakout.brain)

The sample project shows the pipeline for unsupervised discovery of the hypothesized objects. First, an input image is segmented into super-pixels (SP). Second, each SP is connected with its neighbors and close-by SP are assigned into a same object id. Third, the attention energy (Ea) is estimated for each object. Fourth, features are estimated as raw image patches. Fifth, the object features are clustered into a Visual Words to constitute a Working Memory.

Again, `Operation/ObserverMode` property of most of observers switches between visualization modes.

Details are described in [Guides/Image Processing](../guides/improc.md).

![](../img/vision_ex_pong.PNG)
