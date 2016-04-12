# Node and Brain Testing

If you want to test whole brains and classic unit testing is too granular, you can use the **Brain Unit** test library.

Brain Unit enables you to run automated tests on whole brain projects. You can test one or more nodes within one test (fewer is usually better). Brain Unit was inspired by unit testing frameworks for code such as our beloved xUnit and it works in as similar way as possible.

Tests are run using **BrainTestRunner**. You have to manually create brain project in Brain Simulator and run one or more tests on this file. (Projects created programmatically are possible, but not supported for now.)

Each test defines:

- which brain project is to be run,
- maximum number of simulation steps,
- if and when the test should terminate before this maximum step count, and
- assertions that should hold at the end of the run

There are two types of tests that the runner can run:

1. **Code tests** defined by a C# class that are using the **BrainUnit library**
2. **Brain project tests** defined by a brain project that contains exactly one **BrainUnitNode**

To sum it up, Brain Unit consists of three components (located in Testing/Infrastructure in the BrainSimulator solution):

* Runner: `BrainTestRunner`
* `BrainUnit` library that defines abstract classes for the code tests
* `BrainUnitNode`: special testing node for defining the brain project test

## How to create code test in BrainUnit

1. Create a project in Brain Simulator that you'd like to be run in the test and save it to `Sources\Tests\BrainTests\Brains`
1. In VS, create a C# library project called `SomethingBrainTests` (such as `BasicBrainTests`, see Testing/BrainTests in the BrainSimulator solution).
	- Make sure this project's binary is copied to the `BrainTestRunner` (the easiest way is adding it as a Reference)
	- ...or just add the test to a suitable existing `SomethingBrainTests` project
	- Note: It is also possible to add a test directly to a BrainSimulator module (such as `BasicNodes`).
1. Add the test class
	- It should be public (sealed) class derived from `BrainTest`
	- In the constructor set values for `BrainFileName` (to the file name of the BrainSim project from step 1), `MaxStepCount` and optionally `InspectInterval` (see example below)
	- Override the Check method and optionally the `ShouldStop` method
	- In the Check method, you can use xUnit asserts to check some propositions (or you can throw any other exception to indicate failure).
1. Run the runner and see what happens :)

### What happens when you run the test
`BrainTestRunner` runs the project using `MyProjectRunner` for at most `MaxStepCount` steps and each `InspectInterval` steps calls the `ShouldStop` method to find out,  if it should terminate the simulation sooner. When the simulation finishes, it is left in the paused state and its data can be inspected by the `Check` method. 

The **IBrainScan** interface and Brain Unit in general are currently quite limited. If you need something that is not possible, let us know, or try to extend it yourself.

Example test code:

    using GoodAI.Testing.BrainUnit;
    using Xunit;

    public sealed class AccumulatorCanCountToTen : BrainTest
    {
        public AccumulatorCanCountToTen()
        {
            BrainFileName = "accumulator-test.brain";
 
            MaxStepCount = 10;
            InspectInterval = 2;
        }
 
        public override bool ShouldStop(IBrainScan b)
        {
            return b.GetValues(7)[0] > 15;
        }
 
        public override void Check(IBrainScan b)
        {
            Assert.Equal(10, b.GetValues(7)[0]);
        }
    }

## How to create brain project test with BrainUnitNode ##

`BrainUnitNode` enables you to write a test without Visual Studio, but it is ultimately adapted to an equivalent of the code brain test. `BrainUnitNode` is similar to the C# node.  In addition it has some properties for defining the test.

1. Create a brain project and save it to `Sources\Tests\BrainTests\BrainUnitNodeTests`
	- Add `BrainUnitNode` the project and connect it to the output of some other node that you want to evaluate.
	- (Before that, make sure the you have `TestingNodes` module loaded into Brain Simulator.)
	- Set up properties of the `BrainUnitNode`: `MaxStepCount` and optionally the `InspectInterval`
2. Define the `Check` method and optionally the ShouldStop method inside the node's code.
3. Run the runner.
