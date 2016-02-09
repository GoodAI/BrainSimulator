using System;
using CommandLine;
using CommandLine.Text;
using System.Diagnostics;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;

namespace GoodAI.Tests.BrainTestRunner
{
    class Options
    {
        [Option('f', "filter", Required = false, HelpText = "A partial name of the tests that should be run.")]
        public string Filter { get; set; }

        [HelpOption]
        public string GetUsage()
        {
            return HelpText.AutoBuild(this, (HelpText current) => HelpText.DefaultParsingErrorsHandler(this, current));
        }
    }

    internal static class Program
    {
        public static void Main(string[] args)
        {
            var options = new Options();
            if (!Parser.Default.ParseArguments(args, options))
                Console.WriteLine(options.GetUsage());

            ConfigureTypeMap();

            var testRunner = new TestRunner(new TestDiscoverer(options.Filter), new TestReporter());

            testRunner.Run();

            if (Debugger.IsAttached)
            {
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
        }

        private static void ConfigureTypeMap()
        {
            TypeMap.InitializeConfiguration<CoreContainerConfiguration>();
            TypeMap.Verify();
        }
    }
}
