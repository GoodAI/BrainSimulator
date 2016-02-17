using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.TypeMapping;
using Xunit;

namespace CoreTests.Memory
{
    public class MemoryBlockTests : CoreTestBase
    {
        public class TestNode : MyWorkingNode
        {
            public MyMemoryBlock<float> Block;

            public override void UpdateMemoryBlocks()
            {
                Block = MyMemoryManager.Instance.CreateMemoryBlock<float>(this);
                Block.Count = 0;
            }
        }

        [Fact]
        public void ReallocatesCustomMemoryBlock()
        {
            TestNode node = GetTestingNode();

            node.Block.IsDynamic = true;
            node.Block.Reallocate(10);

            Assert.Equal(10, node.Block.Count);
        }

        [Fact]
        public void DoesNotAllocateStaticMemoryBlock()
        {
            TestNode node = GetTestingNode();

            Assert.Throws<InvalidOperationException>(() => node.Block.Reallocate(10));
        }

        private static TestNode GetTestingNode()
        {
            var project = new MyProject {SimulationHandler = TypeMap.GetInstance<MySimulationHandler>()};
            project.SimulationHandler.Simulation.IsStepFinished = true;
            var node = project.CreateNode<TestNode>();

            node.UpdateMemoryBlocks();

            MyMemoryManager.Instance.AllocateBlocks(node, false);
            return node;
        }
    }
}
