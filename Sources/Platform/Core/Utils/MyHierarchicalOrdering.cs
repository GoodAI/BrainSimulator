using GoodAI.Core.Nodes;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Core.Utils
{
    public class MyHierarchicalOrdering : IMyOrderingAlgorithm
    {
        public List<MyNode> EvaluateOrder(MyNodeGroup nodeGroup)
        {            
            HashSet<MyNode> nodes = CollectWorkingNodes(nodeGroup);
            HashSet<MyNode> destinations = FindDestinations(nodeGroup, nodes);

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

            return orderedNodes;
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

        private HashSet<MyNode> FindDestinations(MyNodeGroup nodeGroup, HashSet<MyNode> nodes)
        {
            HashSet<MyNode> destinations = new HashSet<MyNode>(nodes);
            
            foreach (MyNode node in nodes)
            {
                for (int i = 0; i < node.InputBranches; i++)                
                {
                    MyConnection connection = node.InputConnections[i];

                    if (connection != null && nodes.Contains(connection.From))
                    {
                        destinations.Remove(connection.From);
                    }
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

            var backwardConnections = new List<MyConnection>();

            for (int i = 0; i < node.InputBranches; i++)
            {
                MyConnection connection = node.InputConnections[i];

                if (connection == null)
                    continue;

                if (connection.IsLowPriority)
                {
                    backwardConnections.Add(connection);
                    continue;
                }

                if (nodes.Contains(connection.From))
                    VisitNode(connection.From, nodes, orderedNodes);
            }            

            orderedNodes.Add(node);

            foreach (MyConnection connection in backwardConnections.Where(connection => nodes.Contains(connection.From)))
                VisitNode(connection.From, nodes, orderedNodes);

            if (node is MyNodeGroup)
                orderedNodes.AddRange(EvaluateOrder(node as MyNodeGroup));
        }       
    }
}
