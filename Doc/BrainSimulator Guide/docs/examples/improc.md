## Image Processing

These sample projects show several [examples](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/tree/master/Vision) to perform [image processing](../guides/improc.md)

### Segmentation
Brain: [Vision/Vision_SLICsegmentation.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_SLICsegmentation.brain)

The sample shows how to segment input image using SLIC algorithm. Note that the input image is first modified to have a square shape. When one clicks on the observer, there is a `Operation/ObserverMode` property that allows to switch between different visualizations (such as segmentation borders, centers XYZ-color space etc.).

![](../img/vision_ex_SLIC.PNG)

---

### Simple Image Pre-Processing

Brain: [Vision/Vision_segmentationForBreakout.brain](https://github.com/KeenSoftwareHouse/BrainSimulatorSampleProjects/blob/master/Vision/Vision_segmentationForBreakout.brain)

The sample project shows the pipeline for unsupervised discovery of the hypothesized objects. First, an input image is segmented into super-pixels (SP). Second, each SP is connected with its neighbors and close-by SP are assigned into a same object id. Third, the attention energy (Ea) is estimated for each object. Fourth, features are estimated as raw image patches. Fifth, the object features are clustered into a Visual Words to constitute a Working Memory.

Again, `Operation/ObserverMode` property of most of observers switches between visualization modes.

![](../img/vision_ex_pong.PNG)
