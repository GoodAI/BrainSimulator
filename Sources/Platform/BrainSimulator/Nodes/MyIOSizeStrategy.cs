using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Graph;
using Graph.Compatibility;
using BrainSimulatorGUI.NodeView;
using BrainSimulator.Nodes;

namespace BrainSimulatorGUI.Nodes
{
    class MyIOStrategy : ICompatibilityStrategy
    {
        public bool CanConnect(NodeConnector from, NodeConnector to)
        {
            if (from.Node is MyNodeView && to.Node is MyNodeView)
            {
                return to.Node != from.Node && !to.HasConnection;
            }
            else return false;
        }
    }
}
