using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Nodes;
using Xunit;

namespace CoreTests
{
    public class NodeConnectionTests
    {
        // TODO(HonzaS): test the special subclasses of MyNode (Input/Output nodes etc.)
        
        sealed class TestNode : MyWorkingNode
        {
            public TestNode()
            {
                InputBranches = 1;
                OutputBranches = 1;
            }

            public override void UpdateMemoryBlocks()
            {
            }
        }

        [Fact]
        public void Connects()
        {
            var node1 = new TestNode();
            var node2 = new TestNode();

            var connection = new MyConnection(node1, node2, 0, 0);
            connection.Connect();

            Assert.Equal(node1.OutputConnections[0].First().To, node2);
            Assert.Equal(node2.InputConnections[0].From, node1);
        }

        [Fact]
        public void Disconnects()
        {
            var node1 = new TestNode();
            var node2 = new TestNode();

            var connection = new MyConnection(node1, node2, 0, 0);
            connection.Connect();

            connection.Disconnect();
            Assert.Empty(node1.OutputConnections[0]);
            Assert.Null(node2.InputConnections[0]);
        }

        [Fact]
        public void KeepsConnectionsWhenIncreasingOutputBranches()
        {
            var node1 = new TestNode();
            var node2 = new TestNode();

            var connection = new MyConnection(node1, node2, 0, 0);
            connection.Connect();

            node1.OutputBranches = 2;
            
            Assert.Equal(node1.OutputConnections[0].First().To, node2);
            Assert.Equal(node2.InputConnections[0].From, node1);

            Assert.Empty(node1.OutputConnections[1]);
        }

        [Fact]
        public void KeepsConnectionsWhenIncreasingInputBranches()
        {
            var node1 = new TestNode();
            var node2 = new TestNode();

            var connection = new MyConnection(node1, node2, 0, 0);
            connection.Connect();

            node2.InputBranches = 2;
            
            Assert.Equal(node1.OutputConnections[0].First().To, node2);
            Assert.Equal(node2.InputConnections[0].From, node1);

            Assert.Null(node2.InputConnections[1]);
        }

        [Fact]
        public void DisconnectsWhenDecreasingOutputBranches()
        {
            var node1 = new TestNode();
            var node2 = new TestNode();

            var connection = new MyConnection(node1, node2, 0, 0);
            connection.Connect();

            node1.OutputBranches = 0;
            
            Assert.Empty(node1.OutputConnections);
            Assert.Null(node2.InputConnections[0]);
        }

        [Fact]
        public void DisconnectsWhenDecreasingInputBranches()
        {
            var node1 = new TestNode();
            var node2 = new TestNode();

            var connection = new MyConnection(node1, node2, 0, 0);
            connection.Connect();

            node2.InputBranches = 0;
            
            Assert.Empty(node1.OutputConnections[0]);
            Assert.Empty(node2.InputConnections);
        }
    }
}
