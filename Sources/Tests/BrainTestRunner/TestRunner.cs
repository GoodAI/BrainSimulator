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
using Xunit.Sdk;

namespace GoodAI.Tests.BrainTestRunner
{
    internal class InvalidTestException : Exception
    {
        public InvalidTestException(string message) : base(message) { }
    }

    internal class TestRunner
    {
        private readonly TestDiscoverer m_discoverer;
        private readonly TestReporter m_reporter;
        private readonly MyProjectRunner m_projectRunner;

        private readonly string DefaultBrainFilePath = @"../../../BrainTests/Brains/";

        public TestRunner(TestDiscoverer discoverer, TestReporter reporter)
        {
            m_discoverer = discoverer;
            m_reporter = reporter;
            m_projectRunner = new MyProjectRunner(MyLogLevel.WARNING);  // TODO: make it configurable
        }

        public void Run()
        {
            foreach (BrainTest test in m_discoverer.FindTests())
            {
                EvaluateTest(test, m_projectRunner);
            }

            m_reporter.Conclude();

            m_projectRunner.Shutdown();
        }

        private void EvaluateTest(BrainTest test, MyProjectRunner projectRunner)
        {
            try
            {
                OpenProject(test, projectRunner);

                ValidateTest(test);

                m_reporter.StartTest(test);
                
                RunTest(test, m_projectRunner);

                m_reporter.AddPass(test);
            }
            catch (InvalidTestException e)
            {
                m_reporter.AddInvalidTest(test, e);
            }
            catch (BrassertFailedException e)
            {
                m_reporter.AddFail(test, e);
            }
            catch (XunitException e) // TODO: handle specificly AssertActualExpectedException
            {
                m_reporter.AddFail(test, e);
            }
            catch (Exception e)
            {
                m_reporter.AddCrash(test, e);
            }
        }

        private void OpenProject(BrainTest test, MyProjectRunner projectRunner)
        {
            var brainUnitNodeTest = test as BrainUnitNodeTest;
            if (brainUnitNodeTest != null)  // TODO: solve using polymorphism
            {
                brainUnitNodeTest.Initialize(projectRunner);
            }
            else
            {
                projectRunner.OpenProject(FindBrainFile(test));
            }
        }

        private static void RunTest(BrainTest test, MyProjectRunner projectRunner)
        {
            var brainScan = new BrainScan(projectRunner);
            var step = new StepChecker();

            try
            {
                while (true)
                {
                    projectRunner.RunAndPause(GetIterationStepCount(test, projectRunner.SimulationStep));

                    step.AssertIncreased(projectRunner.SimulationStep);

                    if ((projectRunner.SimulationStep >= test.MaxStepCount) || ShouldStop(test, brainScan))
                        break;
                }

                test.Check(brainScan);
            }
            finally
            {
                //MyLog.WARNING.WriteLine("  simulation step: {0}", projectRunner.SimulationStep);  // TODO(Premek): pass simulation step to the reporter

                projectRunner.Reset();
            }
        }

        private static bool ShouldStop(BrainTest test, IBrainScan brainScan)
        {
            try
            {
                if (test.ShouldStop(brainScan))
                    return true;
            }
            // assert failues in ShouldStop mean "don't stop yet" (allow to use same asserts as in Check())
            catch (XunitException) { } 
            catch (BrassertFailedException) { }

            return false;
        }

        private void ValidateTest(BrainTest test)
        {
            if (string.IsNullOrEmpty(test.BrainFileName))
                throw new InvalidTestException("Missing brain file name in the test.");

            FindBrainFile(test);  // ignore result for now

            if (test.MaxStepCount < 1)
                throw new InvalidTestException("Invalid MaxStepCount: " + test.MaxStepCount);
        }

        private string FindBrainFile(BrainTest test)
        {
            string brainFileName = test.BrainFileName;

            if (Path.IsPathRooted(brainFileName))
            {
                if (!File.Exists(brainFileName))
                    throw new InvalidTestException("Brain file not found: " + brainFileName);
                
                return brainFileName;
            }

            string defaultPath = Path.GetFullPath(Path.Combine(DefaultBrainFilePath, brainFileName));
            if (File.Exists(defaultPath))
                return defaultPath;

            // try also relative path
            if (File.Exists(Path.GetFullPath(brainFileName)))
                return Path.GetFullPath(brainFileName);

            throw new InvalidTestException("Brain file not found: " + defaultPath);  // complain about the default path
        }

        private static uint GetIterationStepCount(BrainTest test, uint simulationStep)
        {
            int inspectInterval = test.InspectInterval;
            if (inspectInterval < 1)
                throw new InvalidTestException("Invalid inspect interval: " + inspectInterval);

            return (uint)Math.Min(inspectInterval, test.MaxStepCount - simulationStep);   // limit to remaining steps
        }

        private class StepChecker
        {
            private uint m_stepBefore = 0;

            public void AssertIncreased(uint step)
            {
                if (step == m_stepBefore)
                    throw new InvalidOperationException("Step did not increase: simulation step canceled?");

                m_stepBefore = step;
            }
        }
    }
}
