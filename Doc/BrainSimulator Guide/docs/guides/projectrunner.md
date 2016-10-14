# How to use MyProjectRunner

MyProjectRunner is an alternative to BrainSimulator GUI - you can use it to operate with projects without using GUI.
That is especially handy if you want to:

- Edit 10s or 100s of nodes at once
- Try multiple values for one parameter and watch how it will affect results
- Run project for specific number of steps - then make some changes - run project again. And again
- Create multiple versions of one project
- Get information about value changes during the simulation
- Increase speed of simulation (no GUI)

## Current capabilities
- Print info (`DumpNodes`)
- Open/save/run projects (`OpenProject`, `SaveProject`, `RunAndPause`, `Shutdown`)
- Edit node and task parameters (`Set`)
- Edit multiple nodes at once (`Filter`, `GetNodesOfType`)
- Get or set memory block values (`GetValues`, `SetValues`)
- Track specific memory block values, export data or plot them (`TrackValue`, `Results`, `SaveResults`, `PlotResults`)

## Simple example
``` csharp
MyProjectRunner runner = new MyProjectRunner();
runner.OpenProject(@"C:\Users\johndoe\Desktop\Breakout.brain");
runner.DumpNodes();
runner.RunAndPause(1000);
float[] data = runner.GetValues(24);
MyLog.INFO.WriteLine(data);
runner.Shutdown();
```

## More advanced example

``` csharp
// Program tries different combinations of parameters for two nodes, computes average values for multiple runs, log results and saves them to file.

MyProjectRunner runner = new MyProjectRunner(MyLogLevel.WARNING);
runner.OpenProject(@"C:\Users\johndoe\Desktop\test.brain");
float iterations = 250;

List<Tuple<int, int, float, float>> results = new List<Tuple<int, int, float, float>>();
runner.Set(6, "OutputSize", 32);

for (int symbolSize = 512; symbolSize <= 8192; symbolSize *= 2)
{
   for (int binds = 20; binds <= 50; binds += 5)
   {
        float okSum = 0;
        runner.Set(7, "Binds", binds);
        runner.Set(7, "SymbolSize", symbolSize);
        for (int i = 0; i < iterations; ++i)
        {
            runner.RunAndPause(1, 10);
            float okDot = runner.GetValues(8)[0];
            okSum += okDot;
            runner.Reset();
            if ((i + 1) % 10 == 0)
                MyLog.WARNING.Write('.');
        }
        MyLog.WARNING.WriteLine();
        float wrongSum = 1;
        MyLog.WARNING.WriteLine("Results:" + symbolSize + "@" + binds + " => " + okSum / iterations + " / " + wrongSum / iterations);
        results.Add(new Tuple<int, int, float, float>(symbolSize, binds, okSum / iterations, wrongSum / iterations));
    }
}

File.WriteAllLines(@"C:\Users\johndoe\Desktop\results.txt", results.Select(n => n.ToString().Substring(1, n.ToString().Length - 2)));
runner.Shutdown();
```

Complete documentation for MyProjectRunner API follows.

---
## T: MyProjectRunner

 Alternative to BrainSimulator GUI used to run brains through scripting 


---
### M: DumpNodes

 Prints info about nodes to DEBUG 


---
### M: Filter(FilterFunc)

 Filter all nodes in project recursively. Returns list of nodes, for which the filter function returned True. 

|Name | Description |
|-----|------|
|filterFunc |User-defined function for filtering|
Returns: Node list


---
### M: GetNodesOfType(Type)

 Returns list of nodes of given type 

|Name | Description |
|-----|------|
|type |Node type|
Returns: Node list


---
### M: GetTaskByType(MyWorkingNode, Type)

 Return task of given type from given node 

|Name | Description |
|-----|------|
|node |Node|
|type |Type of task|
Returns: Task


---
### M: GetValues(Int32, String)

 Returns float array of value from memory block of given node 

|Name | Description |
|-----|------|
|nodeId |Node ID|
|blockName |Memory block name|
Returns: Retrieved values


---
### M: SetValues(Int32, Single[], String)

 Set values of memory block 

|Name | Description |
|-----|------|
|nodeId |Node ID|
|values |Values to be set|
|blockName |Name of memory block|


---
### M: Shutdown

 Shutdown the runner and the underlaying simulation infrastructure 


---
### M: OpenProject(String)

 Loads project from file 

|Name | Description |
|-----|------|
|path |Path to .brain/.brainz file|


---
### M: SaveProject(String)

 Saves project to given path 

|Name | Description |
|-----|------|
|path |Path for saving .brain/.brainz file|


---
### M: Set(Int32, String, Object)

 Sets property of given node. Support Enums - enter enum value as a string 

|Name | Description |
|-----|------|
|nodeId |Node ID|
|propName |Property name|
|value |Value to be set|


---
### M: Set(Int32, Type, String, Object)

 Sets property of given task. Support Enums 

|Name | Description |
|-----|------|
|nodeId |Node ID|
|taskType |Task type|
|propName |Property name|
|value |New property value|


---
### M: TrackValue(Int32, String, Int32, UInt32)

 Track a value 

|Name | Description |
|-----|------|
|nodeId |Node ID|
|blockName |Memory block name|
|blockOffset |Offset in given memory block|
|trackInterval |Track value each x steps|
Returns: Result ID


---
### M: Results

 Returns hashtable with results (list of float arrays) 
Returns: Results


---
### M: SaveResults(Int32, String)

 Save result to a file 

|Name | Description |
|-----|------|
|resultId |Result ID|
|outputPath |Path to file in C# format, e.g. C:\path\to\file|


---
### M: PlotResults(Int32[], String, String[], Int32[])

 Plot results to a file 

|Name | Description |
|-----|------|
|resultIds |IDs of results|
|outputPath |Path to file in gnuplot format, e.g. C:/path/to/file|
|lineTitles |Titles of the lines|
|dimensionSizes |Sizes of plot dimensions|


---
### M: RunAndPause(UInt32, UInt32)

 Runs simulation for a given number of steps. Simulation will be left in PAUSED state after this function returns, allowing to inspect content of memory blocks and then perhaps resume the simulation by calling this function again. 

|Name | Description |
|-----|------|
|stepCount |Number of steps to perform|
|reportInterval |Step count between printing out simulation info (e.g. speed)|


---
### M: Reset

 Stops the paused simulation and flushes memory 


---
## T: FilterFunc

 Definition for filtering function 

|Name | Description |
|-----|------|
|node |Node, which is being processed for filtering|
Returns: None


---