# How to create a module with source dependencies

Follow this guide to bootstrap a new module that references the BrainSimulator project instead of the binaries.

## Choose a module name

For this tutorial, we'll use **TestModule** as the name of the module and **MyCompany** as the base namespace of the module. All the names in the following steps are derived from these names.

## Get the sources

1. Clone the [BrainSimulator](https://github.com/GoodAI/BrainSimulator.git) project repository.
2. Clone the [NewModuleWithSourceDeps](https://github.com/GoodAI/NewModuleWithSourceDeps.git) into *BrainSimulator*\Sources\Modules\\**TestModule**

The Node project is located in **TestModule**\Module, the CUDA kernels are in **TestModule**\Cuda

## Rename the module components

1. Open the NewModuleWithSourceDeps solution in VS and rename the solution to **TestModule**
2. Rename the NewModuleWithSourceDeps project to **TestModule**
3. Rename the NewModuleWithSourceDepsCuda project to **TestModuleCuda**
4. In the **TestModule** project, find **NewModuleNode.cs** and rename it to **TestNode.cs**
5. Open the file and change line 13 to:
   
	`namespace MyCompany.Modules.TestModule`

6. Rename the **NewModuleNode** class to **TestNode** and **NewModuleTask** to **TestTask**
7. In **TestModule**\conf\nodes.xml set the type to **MyCompany.Modules.TestModule.TestNode**. It should look like this:

```xml
<?xml version="1.0" encoding="utf-8" ?>
<Configuration RootNamespace="MyCompany.Modules">
	<KnownNodes>
   		<Node type="MyCompany.Modules.NewModuleWithSourceDeps.NewModuleNode" CanBeAdded="true"/>
  	</KnownNodes>
</Configuration>
```
	

## Build and test the module

1. Rebuild the **TestModule** project. VS doesn't build the module project with BrainSimulator when you run it, because it's not a dependency. You need to do this manually.
2. Set the BrainSimulator project as the startup project and then run/debug.

You should be able to load the **TestNode** in the running instance now.