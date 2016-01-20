using System;
using GoodAI.Platform.Core.Configuration;
using GoodAI.TypeMapping;

namespace GoodAI.CoreRunner
{
    class Program
    {
        static void Main(string[] args)
        {
            TypeMap.InitializeConfiguration<CoreContainerConfiguration>();
            TypeMap.Verify();

            ExampleExperiment.Run(args);
        }
    }
}
