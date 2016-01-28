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
        private readonly TestNode m_node1;
        private readonly TestNode m_node2;
        private readonly MyConnection m_connection;
        // TODO(HonzaS): test the special subclasses of MyNode (Input/Output nodes etc.)

        public NodeConnectionTests()
        {
            m_node1 = new TestNode();
            m_node2 = new TestNode();

            m_connection = new MyConnection(m_node1, m_node2, 0, 0);
            m_connection.Connect();
        }
        
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
            Assert.Equal(m_node1.OutputConnections[0].First().To, m_node2);
            Assert.Equal(m_node2.InputConnections[0].From, m_node1);
        }

        [Fact]
        public void Disconnects()
        {
            m_connection.Disconnect();
            Assert.Empty(m_node1.OutputConnections[0]);
            Assert.Null(m_node2.InputConnections[0]);
        }

        [Fact]
        public void KeepsConnectionsWhenIncreasingOutputBranches()
        {
            m_node1.OutputBranches = 2;
            
            Assert.Equal(m_node1.OutputConnections[0].First().To, m_node2);
            Assert.Equal(m_node2.InputConnections[0].From, m_node1);

            Assert.Empty(m_node1.OutputConnections[1]);
        }

        [Fact]
        public void KeepsConnectionsWhenIncreasingInputBranches()
        {
            m_node2.InputBranches = 2;
            
            Assert.Equal(m_node1.OutputConnections[0].First().To, m_node2);
            Assert.Equal(m_node2.InputConnections[0].From, m_node1);

            Assert.Null(m_node2.InputConnections[1]);
        }

        [Fact]
        public void DisconnectsWhenDecreasingOutputBranches()
        {
            m_node1.OutputBranches = 0;
            
            Assert.Empty(m_node1.OutputConnections);
            Assert.Null(m_node2.InputConnections[0]);
        }

        [Fact]
        public void DisconnectsWhenDecreasingInputBranches()
        {
            m_node2.InputBranches = 0;
            
            Assert.Empty(m_node1.OutputConnections[0]);
            Assert.Empty(m_node2.InputConnections);
        }
    }
}
