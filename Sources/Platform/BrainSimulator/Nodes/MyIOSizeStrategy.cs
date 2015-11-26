using GoodAI.BrainSimulator.NodeView;
using Graph;
using Graph.Compatibility;

namespace GoodAI.BrainSimulator.Nodes
{
    class MyIOStrategy : ICompatibilityStrategy
    {
        public bool CanConnect(NodeConnector from, NodeConnector to)
        {
            if (from is NodeInputConnector)
            {
                NodeConnector temp = to;
                to = from;
                from = temp;
            }

            var fromNode = from.Node as MyNodeView;
            var toNode = to.Node as MyNodeView;
            if (fromNode != null && toNode != null)
            {

                return fromNode != toNode && !to.HasConnection &&
                       toNode.Node.AcceptsConnection(fromNode.Node, (int) from.Item.Tag, (int) to.Item.Tag);
            }
            else return false;
        }
    }
}
