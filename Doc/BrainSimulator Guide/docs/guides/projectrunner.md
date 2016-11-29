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

## How to use it

If you want to use `MyProjectRunner` you can either create your own C# project or you can use our `BrainSimulator` solution placed in `Sources` folder. It contains `CoreRunner` project and its `ExampleExperiment` file. You can easily edit the `Run` method to try `MyProjectRunner`.

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
