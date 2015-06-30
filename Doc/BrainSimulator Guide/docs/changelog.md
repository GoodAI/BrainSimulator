TODO
obrazek persistence, custom execution plan

* 6/9/2015
    - ```YAXCustomSerializer(typeof(MyPathSerializer))]``` parameter annotation added

* 3/2015
    - custom_nodes.xml replaced with conf/nodes.xml

* 11/5/2014
    - cyclic update of memory blocks before simulation is started. BS will update all memory blocks until no change in size happens (up to 20 attempts). This should solve most cyclic project issues. However, there must be an “end node” in your model.
    - histogram memory block observer.
    - gate node for online interpolation between two different signals. (under transforms menu)
    - report interval selection. It sets interval between two reports from the running simulation (observer rendering etc.)

* 10/23/2014
    - zoom to fit button
    - zoom and position attributes are now saved into the project (for each group separately)
    - copy & paste nodes

* 10/20/2014
    - basic import of projects (an imported project is placed more or less under your current project)

* 10/14/2014
    - tasks can be ordered by adding Order parameter to MyTaskInfo attribute of the class (i.e. [MyTaskInfo(Order=int value)]). By default the order is zero. (The feature is useful i.e. for inheritance of nodes.)

* 9/26/2014
    - folder structure for saving persistable memory blocks changed (not backward compatible, but older data can be still used);
    - state of persistable memory blocks of an individual nodes can be loaded (see icon in top right corner of Node Properties window;
    - all output blocks are persistable by default (cannot be changed) - it is needed for consistent loading
    - blue channel in RedGreenScale observer indicates infinities and NaN (magenta = -infinity, cyan = infinity, blue = NaN)

* 9/4/2014
    - MinValueHint & MaxValueHint properties in the memory block class. Observers will be configured according to them (if set).
    - Due to some inconsistencies when deriving from nodes which already have some input & output blocks defined it is necessary to annotate all input and output blocks with Order attribute (e.g. [MyInputBolock(0)], [MyInputBolock(1)]…). It is mandatory only for derived nodes and their super nodes. Some for output blocks.
    - MyWorld class now contains virtual Cleanup() method which is called right after the simulation is stopped. It is empty by default. You may put some resource cleanup stuff inside if needed.
    - Dedicated RandDevice per GPU. Use MyKernelFactory.GetRandDevice() method (no prepare method is needed).- Kernels update: Reduction kernel supports offsets for input & output (MBa). Image convolution for 3×3 kernels (DF). Letter rendering in observers (PH).
    - Nodes update: Filter2D contains Gaussian blur and Sobel edge detector
    - New worlds: MNISTWorld (MV), MovingObjectsWorld (DF), ArmWorld (KK)

* 8/25/2014
    - Node API refactoring complete. From now on, nodes can have multiple input & output points. Use [MyInputBlock] & [MyOutputBlock] for definitions (more will be in the guide).
    - Side tool bar per user configuration (View->Configure Node Selection, Ctrl+L), so the user_nodes.xml file is not needed anymore (every node should be in custom_nodes.xml).
    - Automatic icon creation. PLEASE REMOVE ALL REFERENCES TO no_image.png ICON IN CONFIG FILES.
    - All above applies for world nodes as well. World nodes custom observers are also allowed.- THERE IS NO BACKWARD COMPATIBILITY FOR .brain PROJECTS. Y’ll need to manually alter the old project to get it working (ask me for details).

* 8/14/2014
    - Plot observer for memory blocks (thanks to Pascal) is now available. You can watch temporal change of any selected value within memory block.7/29/2014- Parallel reduction node
    - Memory blocks persistence (Save/Load of network state during paused simulation). Persistent memory block must be annotated with MyPersistable attribute.

* 7/28/2014
    - Topological execution of nodes (all nodes are executed according to their “topological order” from start to end), i.e. there shouldn’t be any delay in a transform chain
    - Per user node config files.
    - Node observers can be added during simulation
    - Various observer code changes, including SimulationStep accessibility within observer.

* 7/23/2014
    - Transforms are now located in their own menu (see transform_test.brain example project in the BrainSimulatorGUI project root)
    - MyTransform & MyTransformGroup introduced
    - MyAccumulator introduced (replacement of StateTransform and DelayedCopyNode)
    - MyFilter2D & MyResize2D introduced (replacement for MyScaleImageNode and part of MyBasicTransform)
    - Various transform nodes introduced
    - Fill & Copy methods for memory blocks (memset, memcpy)
    - More kernels in one .cu file
    - SimluationStep property in MyTask
    - Obsolete annotation for nodes

* Older changes 
    - Kernel Reloading! You can rebuild kernels outside of the running GUI and reload them.
    - MyLog level setting in console view
    - Observer view properties moved to property grid on the left side
    - …