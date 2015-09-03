# How to create a module with source dependencies

Follow this guide to bootstrap a new module that references the BrainSimulator project instead of the binaries.

## Choose a module name

For this tutorial, we'll use **TestModule** as the name of the module. All the names in the following steps are derived from this name.

## Get the sources

1. Clone the [BrainSimulator](https://github.com/GoodAI/BrainSimulator.git) project repository.
2. Clone the [NewModuleWithSourceDeps](https://github.com/GoodAI/NewModuleWithSourceDeps.git) into *BrainSimulator*\Sources\Modules\\**TestModule**

The Node project is located in **TestModule**\Module, the CUDA kernels are in **TestModule**\Cuda

## Rename the module components

1. Open the NewModuleWithSourceDeps solution in VS and rename the solution to **TestModule**
2. Rename the NewModuleWithSourceDeps project to **TestModule**
3. Rename the NewModuleWithSourceDepsCuda project to **TestModuleCuda**
4. In the **TestModule** project, open the Node class
	* Rename the file to **MyTestNode.cs**
	* Rename the namespace to **GoodAI.Modules.TestModule**
	* Rename the Node and Task classes to **MyTestNode** and **MyTestTask** respectively
8. In **TestModule**\conf\nodes.xml set the type to **GoodAI.Modules.TestModule.MyTestNode**

## Build and test the module

9. Rebuild the **TestModule** project. The project is not required to be rebuilt when BS is built, so it must be done manually
10. Run/debug the BrainSimulator project (tip: set it as the startup project)

You should be able to load the **TestNode** in the running instance now.