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

namespace GoodAI.Tests.BrainTestRunner
{
    class AssertExperiment
    {
        public static void Run(string[] args)
        {
            // -clusterid $(Cluster) -processid $(Process) -brain Breakout.brain -factor 0.5

            int clusterId = 0;
            int processId = 0;
            double discountFactor = 0.6;
            string brainFilePath = "";
            OptionSet options = new OptionSet()
                .Add("clusterid=", v => clusterId = Int32.Parse(v))
                .Add("processid=", v => processId = Int32.Parse(v))
                .Add("factor=", v => discountFactor = Double.Parse(v, CultureInfo.InvariantCulture))
                .Add("brain=", v => brainFilePath = Path.GetFullPath(v));

            try
            {
                options.Parse(Environment.GetCommandLineArgs().Skip(1));
            }
            catch (OptionException e)
            {
                MyLog.ERROR.WriteLine(e.Message);
            }

            if (string.IsNullOrEmpty(brainFilePath))
                brainFilePath = Path.GetFullPath("../../../BrainTests/Brains/accumulator-test.brain");

            MyProjectRunner runner = new MyProjectRunner(MyLogLevel.DEBUG);
            StringBuilder result = new StringBuilder();

            runner.OpenProject(brainFilePath);
            runner.DumpNodes();

            runner.RunAndPause(10, 1);

            float[] outputData = runner.GetValues(7);  // accumulator node (Output)

            MyLog.DEBUG.WriteLine("Expected: 10, Actual: {0}", outputData[0]);

            runner.Reset();

            //string resultFilePath = @"res." + clusterId.ToString() + "." + processId.ToString() + ".txt";
            //File.WriteAllText(resultFilePath, result.ToString());

            runner.Shutdown();
            return;
        }
    }
}
