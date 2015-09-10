using GoodAI.BrainSimulator.NodeView;
using Graph;
using Graph.Compatibility;

namespace GoodAI.BrainSimulator.Nodes
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
