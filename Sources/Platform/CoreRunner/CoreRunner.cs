using GoodAI.Modules.NeuralNetwork.Tasks;
using GoodAI.Core.Utils;
using GoodAI.Core.Execution;
using NDesk.Options;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Globalization;

namespace GoodAI.CoreRunner
{
    class Program
    {
        static void Main(string[] args)
        {
            // -clusterid $(Cluster) -processid $(Process) -brain Breakout.brain -factor 0.5

            int clusterId = 0;
            int processId = 0;
            double discountFactor = 0.6;
            string breakoutBrainFilePath = "";
            OptionSet options = new OptionSet()
                .Add("clusterid=", v => clusterId = Int32.Parse(v))
                .Add("processid=", v => processId = Int32.Parse(v))
                .Add("factor=", v => discountFactor = Double.Parse(v, CultureInfo.InvariantCulture))
                .Add("brain=", v => breakoutBrainFilePath = Path.GetFullPath(v));

            try
            {
                options.Parse(Environment.GetCommandLineArgs().Skip(1));
            }
            catch (OptionException e)
            {
                MyLog.ERROR.WriteLine(e.Message);
            }

            MyProjectRunner runner = new MyProjectRunner(MyLogLevel.DEBUG);
            StringBuilder result = new StringBuilder();
            runner.OpenProject(breakoutBrainFilePath);
            runner.DumpNodes();
            runner.Save(23, true);
            for (int i = 0; i < 5; ++i )
            {
                runner.Run(1000, 100);
                float[] data = runner.GetValues(23, "Bias");
                MyLog.DEBUG.WriteLine(data[0]);
                MyLog.DEBUG.WriteLine(data[1]);
                result.AppendFormat("{0}: {1}, {2}", i, data[0], data[1]);
                runner.Stop();
                runner.Set(23, typeof(MyQLearningTask), "DiscountFactor", discountFactor);
                runner.Run(1000, 300);
                data = runner.GetValues(23, "Bias");
                MyLog.DEBUG.WriteLine(data[0]);
                MyLog.DEBUG.WriteLine(data[1]);
                result.AppendFormat(" --- {0}, {1}", data[0], data[1]).AppendLine();
                runner.Stop();
            }

            string resultFilePath = @"res." + clusterId.ToString() + "." + processId.ToString() + ".txt";
            File.WriteAllText(resultFilePath, result.ToString());
            string brainzFilePath = @"state." + clusterId.ToString() + "." + processId.ToString() + ".brainz";
            runner.SaveProject(brainzFilePath);

            runner.Quit();
            return;
        }
    }
}

//# How to use MyProjectRunner

//MyProjectRunner is an alternative to BrainSimulator GUI - you can use it to operate with projects without using GUI. That is especially handy if you want to:
//* Edit 10s or 100s of nodes at once
//* Try multiple values for one parameter and watch how it will affect results
//* Run project for specific number of steps - then make some changes - run project again. And again
//* Create multiple versions of one project
//* Get information about value changes during the simulation
//* Increase speed of simulation (no GUI)

//All important methods should have explanatory comments

//## Current capabilities
//* Print info (DumpNodes)
//* Open/save/run projects (OpenProject, SaveProject, Run, Quit)
//* Edit node and task parameters (Set)
//* Edit multiple nodes at once (Filter, GetNodesOfType)
//* Get or set memory block values (GetValues, SetValues)
//* Track specific memory block values, export data or plot them (TrackValue, Results, SaveResults, PlotResults)

//## Simple example:

//MyProjectRunner runner = new MyProjectRunner(MyLogLevel.DEBUG);
//runner.OpenProject(@"C:\Users\johndoe\Desktop\Breakout.brain");
//runner.DumpNodes();
//runner.Run(1000, 100);
//float[] data = runner.GetValues(24, "Output");
//MyLog.INFO.WriteLine(data);
//runner.Quit();
//Console.ReadLine();

//## More advanced example.
//Program tries different combinations of parameters for two nodes, computes average values for multiple runs, log results and saves them to file.
//:

//MyProjectRunner runner = new MyProjectRunner(MyLogLevel.WARNING);
//runner.OpenProject(@"C:\Users\johndoe\Desktop\test.brain");
//float iterations = 250;

//List<Tuple<int, int, float, float>> results = new List<Tuple<int, int, float, float>>();
//runner.Set(6, "OutputSize", 32);

//for (int symbolSize = 512; symbolSize <= 8192; symbolSize *= 2)
//{
//   for (int binds = 20; binds <= 50; binds += 5)
//   {
//        float okSum = 0;
//        runner.Set(7, "Binds", binds);
//        runner.Set(7, "SymbolSize", symbolSize);
//        for (int i = 0; i < iterations; ++i)
//        {
//            runner.Run(1, 10);
//            float okDot = runner.GetValues(8)[0];
//            okSum += okDot;
//            runner.Stop();
//            if ((i + 1) % 10 == 0)
//            {
//                MyLog.WARNING.Write('.');
//            }
//        }
//        MyLog.WARNING.WriteLine();
//        float wrongSum = 1;
//        MyLog.WARNING.WriteLine("Results:" + symbolSize + "@" + binds + " => " + okSum / iterations + " / " + wrongSum / iterations);
//        results.Add(new Tuple<int, int, float, float>(symbolSize, binds, okSum / iterations, wrongSum / iterations));
//    }
//}

//File.WriteAllLines(@"C:\Users\johndoe\Desktop\results.txt", results.Select(n => n.ToString().Substring(1, n.ToString().Length - 2)));

//runner.Quit();
