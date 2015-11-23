using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

using GoodAI.Core.Configuration;
using GoodAI.BrainSimulator.Nodes;
using GoodAI.Modules.Foo.Bar;  // mock namespace

namespace GoodAI.BrainSimulator.Tests
{
    public class NodeCategoryTests
    {
        internal class TestType
        {
        }

        [Fact]
        public void DetectsCategoryFromNamespace()
        {
            var nodeConfig = new MyNodeConfig();
            nodeConfig.NodeType = typeof(TestType);

            Assert.Equal("(GoodAI.BrainSimulator)", CategorySortingHat.DetectCategoryName(nodeConfig));
        }

        [Fact]
        public void DetectsCategoryFromGoodAiModulesNamespace()
        {
            var nodeConfig = new MyNodeConfig();
            nodeConfig.NodeType = typeof(FoobarType);

            Assert.Equal("(Foo)", CategorySortingHat.DetectCategoryName(nodeConfig));
        }
    }
}
