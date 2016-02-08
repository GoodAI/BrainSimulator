using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;

namespace GoodAI.Platform.Core.Nodes
{
    public interface IModelChanges
    {
        void AddNode(MyWorkingNode node);
        void RemoveNode(MyWorkingNode node);

        IEnumerable<MyWorkingNode> AddedNodes { get; }
        IEnumerable<MyWorkingNode> RemovedNodes { get; }
    }

    internal class ModelChanges : IModelChanges
    {
        private readonly List<MyWorkingNode> m_addedNodes;
        private readonly List<MyWorkingNode> m_removedNodes;

        public IEnumerable<MyWorkingNode> AddedNodes { get { return m_addedNodes; } }
        public IEnumerable<MyWorkingNode> RemovedNodes { get { return m_removedNodes; } }

        public ModelChanges()
        {
            m_addedNodes = new List<MyWorkingNode>();
            m_removedNodes = new List<MyWorkingNode>();
        }

        public void AddNode(MyWorkingNode node)
        {
            m_addedNodes.Add(node);
        }

        public void RemoveNode(MyWorkingNode node)
        {
            m_removedNodes.Add(node);
        }
    }

    public interface IModelChanger
    {
        bool ChangeModel(IModelChanges changes);
        MyNode AffectedNode { get; }
    }
}
