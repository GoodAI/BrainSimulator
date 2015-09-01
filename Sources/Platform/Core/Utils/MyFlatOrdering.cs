using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using System.Collections.Generic;

namespace GoodAI.Core.Utils
{
    public interface IMyOrderingAlgorithm
    {
        List<MyNode> EvaluateOrder(MyNodeGroup nodeGroup);
    }

    public class MyFlatOrdering : IMyOrderingAlgorithm
    {
        HashSet<MyNode> m_destinations = new HashSet<MyNode>();
        HashSet<MyNode> m_nodes = new HashSet<MyNode>();
        List<MyNode> m_orderedNodes = new List<MyNode>();

        private int currentOrder;

        public List<MyNode> EvaluateOrder(MyNodeGroup nodeGroup)
        {
            CollectWorkingNodes(nodeGroup);
            FindDestinations();

            if (m_destinations.Count == 0 && m_nodes.Count > 0)
            {
                MyLog.WARNING.WriteLine("Topological ordering failed in node group \"" + nodeGroup.Name + "\"! (no natural destination node of execution graph)");
            }

            //force last node as destination (must be here due possible cycle)
            foreach (MyOutput networkOutput in nodeGroup.GroupOutputNodes)
            {
                if (networkOutput.Input != null)
                {
                    m_destinations.Add(networkOutput.Input.Owner);
                }
            }

            currentOrder = 1;

            foreach (MyNode destNode in m_destinations)
            {
                VisitNode(destNode);
            }

            m_orderedNodes.Clear();
            m_orderedNodes.AddRange(m_nodes);
            m_orderedNodes.Sort((i1, i2) => i1.TopologicalOrder.CompareTo(i2.TopologicalOrder));

            return m_orderedNodes;
        }

        private void CollectWorkingNodes(MyNodeGroup nodeGroup)
        {
            m_nodes.Clear();

            MyNodeGroup.IteratorAction action = delegate(MyNode node)
            {
                if (node is MyWorkingNode)
                {
                    if (node.Parent != null)
                    {
                        node.Sequential = node.Parent.Sequential;
                    }
                    node.TopologicalOrder = -1;
                    m_nodes.Add(node);
                }
            };

            nodeGroup.Iterate(true, action);
        }

        private void FindDestinations()
        {
            m_destinations.Clear();
            m_destinations.UnionWith(m_nodes);

            foreach (MyNode node in m_nodes)
            {
                for (int i = 0; i < node.InputBranches; i++)
                {
                    MyMemoryBlock<float> ai = node.GetInput(i);

                    if (ai != null)
                    {
                        m_destinations.Remove(ai.Owner);
                    }
                }
            }            
        }

        private void VisitNode(MyNode node)
        {
            if (node.TopologicalOrder == -2)
            {
                MyLog.DEBUG.WriteLine("TopoOrdering: cycle detected");
            }
            //skip visited nodes
            if (node.TopologicalOrder != -1) return;

            //mark node as processed
            node.TopologicalOrder = -2;

            for (int i = 0; i < node.InputBranches; i++)
            {
                MyMemoryBlock<float> ai = node.GetInput(i);

                if (ai != null)
                {
                    VisitNode(ai.Owner);
                }
            }

            node.TopologicalOrder = currentOrder;
            currentOrder++;
        }       
    }
}
