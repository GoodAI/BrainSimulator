using System;

namespace GoodAI.Tests.BrainTestRunner
{
    internal static class Program
    {
        public static void Main(string[] args)
        {
            var testRunner = new TestRunner(new TestDiscoverer(), new TestReporter());

            testRunner.Run();
        }
    }
}
