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
        private readonly TestReporter m_reporter;
        private readonly MyProjectRunner m_projectRunner;

        private readonly string DefaultBrainFilePath = @"../../../BrainTests/Brains/";

        public TestRunner(TestReporter reporter)
        {
            m_reporter = reporter;
            m_projectRunner = new MyProjectRunner(MyLogLevel.ERROR);
        }

        public void Run()
        {
            var test = new MyAccumulatorTest();

            EvaluateTest(test);
            EvaluateTest(new MyFailingAccumulatorTest());
            EvaluateTest(test);

            m_reporter.Conclude();

            m_projectRunner.Shutdown();
        }

        private void EvaluateTest(BrainTest test)
        {
            try
            {
                CheckTest(test);

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

        private void RunTest(BrainTest test, MyProjectRunner projectRunner)
        {
            projectRunner.OpenProject(FindBrainFile(test));
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
            }
        }

        private void CheckTest(BrainTest test)
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

        private uint GetIterationStepCount(BrainTest test)
        {
            var inspectInterval = test.InspectInterval;
            if (inspectInterval < 1)
                throw new InvalidTestException("Invalid inspect interval: " + inspectInterval);

            return (uint)inspectInterval;
        }
    }
}
