using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using GoodAI.Core.Utils;
using GoodAI.Platform.Core.Configuration;
using GoodAI.Platform.Core.Nodes;
using GoodAI.TypeMapping;
using Xunit;

namespace CoreTests
{
    public class ContainerConfigurationTests : ContainerConfigurationTestBase
    {
        [Fact]
        public void ResolvesValidator()
        {
            CheckResolveItem<MyValidator>();
        }

        [Fact]
        public void ResolvesExecutionPlanner()
        {
            TypeMap.GetInstance<IMyExecutionPlanner>();
        }

        [Fact]
        public void ResolvesSimulation()
        {
            TypeMap.GetInstance<MySimulation>();
        }

        [Fact]
        public void ResolvesModelChanges()
        {
            TypeMap.GetInstance<IModelChanges>();
        }
    }
}
