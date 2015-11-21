using GoodAI.Core.Utils;
using GoodAI.Core.Execution;
using GoodAI.Modules.Tests;
using GoodAI.Testing.BrainUnit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Tests.BrainTestRunner
{
    internal class InvalidTestExc : Exception
    {
        public InvalidTestExc(string message) : base(message) { }
    }

    internal class TestRunner
    {
        public void Run()
        {
            var test = new MyAccumulatorTest();

            try
            {
                RunTest(test);
            }
            // TODO: make a report
            catch (InvalidTestExc e)
            {
                MyLog.ERROR.WriteLine("Invalid test ({0}): {1}", test.GetType(), e.Message);
            }
            catch (BrassertFailedExc e)
            {
                MyLog.ERROR.WriteLine("Test FAILED ({0}): {1}", test.GetType(), e.Message);
            }
            catch (Exception e)
            {
                MyLog.ERROR.WriteLine("Test crash ({0}): {1}", test.GetType(), e.Message);
            }
        }

        private void RunTest(BrainTest test)
        {
            var projectRunner = new MyProjectRunner();

            CheckTest(test);

            projectRunner.OpenProject(Path.GetFullPath(test.BrainFileName));
            projectRunner.DumpNodes();

            var brainScan = new BrainScan(projectRunner);

            try
            {
                do
                {
                    projectRunner.RunAndPause(GetIterationStepCount(test));

                    if (test.ShouldStop(brainScan))  // TODO(Premek): consider tolerating ShouldStop exceptions (?)
                        break;
                }
                while (projectRunner.SimulationStep < test.MaxStepCount);

                test.Check(brainScan);
            }
            finally
            {
                projectRunner.Reset();  // TODO(Premek): rename to Stop
                projectRunner.Shutdown();
            }
        }

        private void CheckTest(BrainTest test)
        {
            if (string.IsNullOrEmpty(test.BrainFileName))
                throw new InvalidTestExc("Missing brain file name in the test.");

            if (!File.Exists(Path.GetFullPath(test.BrainFileName)))
                throw new InvalidTestExc("Brain file not found.");

            if (test.MaxStepCount < 1)
                throw new InvalidTestExc("Invalid MaxStepCount: " + test.MaxStepCount);
        }

        private uint GetIterationStepCount(BrainTest test)
        {
            var inspectInterval = test.InspectInterval;
            if (inspectInterval < 1)
                throw new InvalidTestExc("Invalid inspect interval: " + inspectInterval);

            return (uint)inspectInterval;
        }
    }
}
