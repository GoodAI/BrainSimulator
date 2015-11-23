using System;

namespace GoodAI.Tests.BrainTestRunner
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            var reporter = new TestReporter();

            var testRunner = new TestRunner(reporter);

            testRunner.Run();
        }
    }
}
