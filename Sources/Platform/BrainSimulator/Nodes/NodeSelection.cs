using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;

namespace GoodAI.BrainSimulator.Nodes
{
    /// <summary>
    /// Represents a selection of MyWorkingNode(s). All nodes must be of the same type, otherwise the selection will be empty.
    /// </summary>
    public class NodeSelection
    {
        public static NodeSelection Empty { get; } = new NodeSelection();

        public List<MyWorkingNode> Nodes { get; } = new List<MyWorkingNode>();

        public List<TaskSelection> Tasks => GatherTasks();

        public bool IsEmpty => !Nodes.Any();

        public int Count => Nodes.Count;

        public NodeSelection(IEnumerable<object> nodes)
        {
            if (nodes == null)
                return;

            var workingNodes = nodes.Select(n => n as MyWorkingNode).Where(n => n != null).ToList();

            var type = workingNodes.FirstOrDefault()?.GetType();

            if (!workingNodes.All(n => (n.GetType() == type)))  // All of the same type or nothing.
                return;

            Nodes = workingNodes;  
        }

        private NodeSelection()
        {
        }

        private List<TaskSelection> GatherTasks()
        {
            return Nodes.First().GetInfo().OrderedTasks.Select(taskInfo => new TaskSelection(taskInfo, Nodes)).ToList();
        }
    }
}