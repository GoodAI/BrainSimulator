using System;

namespace GoodAI.Tests.BrainTestRunner
{
    class Program
    {
        static void Main(string[] args)
        {
            //AssertExperiment.Run(args);

            var testRunner = new TestRunner();

            testRunner.Run();
        }
    }
}
