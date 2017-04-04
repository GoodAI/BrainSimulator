using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CoreTests;
using GoodAI.BrainSimulator.Nodes;
using GoodAI.Core.Nodes;
using GoodAI.Modules.Transforms;
using Xunit;

namespace GoodAI.BrainSimulator.Tests.Nodes
{
    public class NodeSelectionTests : CoreTestBase
    {
        [Fact]
        public void SameNodesAreAccepted()
        {
            var nodeSelection = new NodeSelection(new MyWorkingNode[] { new MyUserInput(), new MyUserInput() });

            Assert.False(nodeSelection.IsEmpty);
            Assert.Equal(2, nodeSelection.Count);
        }

        [Fact]
        public void DifferentNodeTypesYieldEmptySelection()
        {
            var nodeSelection = new NodeSelection(new MyWorkingNode[] { new MyPolynomialFunction(), new MyUserInput() });
            
            Assert.True(nodeSelection.IsEmpty);
        }
    }
}
