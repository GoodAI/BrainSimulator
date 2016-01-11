using GoodAI.Core.Nodes;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Core.Utils
{
    public class MyHierarchicalOrdering : IMyOrderingAlgorithm
    {
        public List<MyNode> EvaluateOrder(MyNodeGroup nodeGroup)
        {
            IList<MyConnection> lowPriorityConnections = new List<MyConnection>();

            HashSet<MyNode> nodes = CollectWorkingNodes(nodeGroup);
            HashSet<MyNode> destinations = FindDestinations(nodeGroup, nodes, ref lowPriorityConnections);

            if (destinations.Count == 0 && nodes.Count > 0)
            {
                MyLog.WARNING.WriteLine("Topological ordering failed in node group \"" + nodeGroup.Name + "\"! (no natural destination node of execution graph)");
            }
           
            List<MyNode> orderedNodes = new List<MyNode>();

            foreach (MyNode destNode in destinations)
            {
                VisitNode(destNode, nodes, orderedNodes);
            }

            int currentOrder = 1;

            foreach (MyNode node in orderedNodes)
            {
                node.TopologicalOrder = currentOrder;
                currentOrder++;
            }

            var edgesChanged = false;
            foreach (MyConnection connection in lowPriorityConnections.Where(
                connection => !connection.From.CheckForCycle(connection.To)))
            {
                connection.IsLowPriority = false;
                edgesChanged = true;
            }
            
            return edgesChanged ? EvaluateOrder(nodeGroup) : orderedNodes;
        }        

        private HashSet<MyNode> CollectWorkingNodes(MyNodeGroup nodeGroup)
        {
            HashSet<MyNode> nodes = new HashSet<MyNode>();            

            MyNodeGroup.IteratorAction action = delegate(MyNode node)
            {
                if (node is MyWorkingNode)
                {
                    node.TopologicalOrder = -1;
                    nodes.Add(node);
                }
            };

            nodeGroup.Iterate(false, action);

            return nodes;
        }

        private HashSet<MyNode> FindDestinations(MyNodeGroup nodeGroup, HashSet<MyNode> nodes, ref IList<MyConnection> lowPriorityConnections)
        {
            HashSet<MyNode> destinations = new HashSet<MyNode>(nodes);
            
            foreach (MyNode node in nodes)
            {
                for (int i = 0; i < node.InputBranches; i++)                
                {
                    MyConnection connection = node.InputConnections[i];

                    if (connection == null)
                        continue;

                    if (connection.IsLowPriority)
                        lowPriorityConnections.Add(connection);
                    else if (nodes.Contains(connection.From))
                        destinations.Remove(connection.From);
                }
            }

            //force nodes before output nodes as destinations (must be here due possible cycle)            
            foreach (MyOutput groupOutput in nodeGroup.GroupOutputNodes)
            {
                if (groupOutput.InputConnections[0] != null)
                {
                    MyNode beforeOutput = groupOutput.InputConnections[0].From;

                    if (nodes.Contains(beforeOutput) && !destinations.Contains(beforeOutput))
                    {
                        destinations.Add(beforeOutput);
                    }
                }
            }            

            return destinations;
        }

        private void VisitNode(MyNode node, HashSet<MyNode> nodes, List<MyNode> orderedNodes)
        {
            if (node.TopologicalOrder == -2 && !orderedNodes.Contains(node))
            {
                MyLog.DEBUG.WriteLine("TopoOrdering: cycle detected");
            }
            //skip visited nodes
            if (node.TopologicalOrder != -1) return;

            //mark node as processed
            node.TopologicalOrder = -2;

            for (int i = 0; i < node.InputBranches; i++)
            {
                MyConnection connection = node.InputConnections[i];

                if (connection == null)
                    continue;

                // Low priority connections are not processed. The nodes will be added to the destinations instead.
                if (connection.IsLowPriority)
                    continue;

                if (nodes.Contains(connection.From))
                    VisitNode(connection.From, nodes, orderedNodes);
            }            

            orderedNodes.Add(node);

            if (node is MyNodeGroup)
                orderedNodes.AddRange(EvaluateOrder(node as MyNodeGroup));
        }       
    }
}
