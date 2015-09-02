using CLIWrapper;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLITester
{
    class CLIProgram
    {
        static void Main(string[] args)
        {
            BSCLI CLI = new BSCLI(MyLogLevel.DEBUG);
            CLI.OpenProject(@"C:\Users\michal.vlasak\Desktop\Breakout.brain");
            CLI.DumpNodes();
            for (int i = 0; i < 5; ++i )
            {
                CLI.Run(1000, 100);
                float[] data = CLI.GetValues(23, "Bias");
                MyLog.DEBUG.WriteLine(data[0]);
                MyLog.DEBUG.WriteLine(data[1]);
                CLI.Stop();
                CLI.Set(23, typeof(MyQLearningTask), "DiscountFactor", 0.6);
                CLI.Run(1000, 300);
                data = CLI.GetValues(23, "Bias");
                MyLog.DEBUG.WriteLine(data[0]);
                MyLog.DEBUG.WriteLine(data[1]);
            }
            CLI.Quit();
            Console.ReadLine();
            return;
        }
    }
}

//# How to use BrainSimCLI

//BrainSimulatorCLI is an alternative to BrainSimulatorGUI - you can use it to operate with projects without using GUI. That is especially handy if you want to:
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

//BSCLI CLI = new BSCLI(MyLogLevel.DEBUG);
//CLI.OpenProject(@"C:\Users\johndoe\Desktop\Breakout.brain");
//CLI.DumpNodes();
//CLI.Run(1000, 100);
//float[] data = CLI.GetValues(24, "Output");
//MyLog.INFO.WriteLine(data);
//CLI.Quit();
//Console.ReadLine();

//## More advanced example.
//Program tries different combinations of parameters for two nodes, computes average values for multiple runs, log results and saves them to file.
//:

//BSCLI CLI = new BSCLI(MyLogLevel.WARNING);
//CLI.OpenProject(@"C:\Users\johndoe\Desktop\test.brain");
//float iterations = 250;

//List<Tuple<int, int, float, float>> results = new List<Tuple<int, int, float, float>>();
//CLI.Set(6, "OutputSize", 32);

//for (int symbolSize = 512; symbolSize <= 8192; symbolSize *= 2)
//{
//   for (int binds = 20; binds <= 50; binds += 5)
//   {
//        float okSum = 0;
//        CLI.Set(7, "Binds", binds);
//        CLI.Set(7, "SymbolSize", symbolSize);
//        for (int i = 0; i < iterations; ++i)
//        {
//            CLI.Run(1, 10);
//            float okDot = CLI.GetValues(8)[0];
//            okSum += okDot;
//            CLI.Stop();
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

//CLI.Quit();

//## Example using all features
//TODO

