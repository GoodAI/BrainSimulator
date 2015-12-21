using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;

namespace GoodAI.BrainSimulator.Utils
{
    public class ProjectState
    {
        public string SerializedProject { get; private set; }
        public string ProjectPath { get; set; }
        public string Action { get; set; }
        public List<int> GraphPanes { get; private set; }
        public int SelectedGraphView { get; set; }
        public string SelectedObserver { get; set; }
        public int SelectedNode { get; set; }

        public ProjectState(string serializedProject)
        {
            if (serializedProject == null)
                throw new ArgumentNullException("serializedProject");

            GraphPanes = new List<int>();
            SerializedProject = serializedProject;
        }

        public override bool Equals(object obj)
        {
            return GetHashCode().Equals(obj.GetHashCode());
        }

        public override int GetHashCode()
        {
            // Two states are the same if the serialized project and path are the same.
            // This is used when a new state is saved - if it equals the previous one, it is discarded.
            // Some information might therefore be lost - add the properties here if necessary.
            unchecked
            {
                int hash = SerializedProject.GetHashCode()*31 + 17;
                if (ProjectPath != null)
                    hash = hash*31 + ProjectPath.GetHashCode();

                return hash;
            }
        }
    }

    /// <summary>
    /// The manager keeps historical and future states for the undo capability of the BrainSim UI.
    /// </summary>
    public class UndoManager
    {
        private uint HistorySize { get; set; }

        // The history. m_undoStates. First() is the oldest item in the history. Last() is the current project state.
        private readonly LinkedList<ProjectState> m_undoStates = new LinkedList<ProjectState>();

        // The future. The item on the top of the stack is the next step.
        private readonly Stack<ProjectState> m_redoStates = new Stack<ProjectState>();

        public UndoManager(uint historySize)
        {
            HistorySize = historySize;
        }

        public void SaveState(ProjectState state)
        {
            if (HistorySize == 0)
                return;

            // Clear the redo stack, an action means we're throwing away the future.
            m_redoStates.Clear();

            // HistorySize + 1 (1 for the current step).
            while (m_undoStates.Count >= HistorySize + 1)
                m_undoStates.RemoveFirst();

            m_undoStates.AddLast(state);
        }

        public ProjectState Undo()
        {
            if (!CanUndo())
                return null;

            ProjectState currentState = m_undoStates.Last();

            m_undoStates.RemoveLast();

            m_redoStates.Push(currentState);

            return m_undoStates.Last();
        }

        public ProjectState Redo()
        {
            if (!CanRedo())
                return null;

            ProjectState nextState = m_redoStates.Pop();

            m_undoStates.AddLast(nextState);

            return nextState;
        }

        public void Clear()
        {
            m_undoStates.Clear();
            m_redoStates.Clear();
        }

        public bool CanUndo()
        {
            // The last state is "current".
            return m_undoStates.Count > 1;
        }

        public bool CanRedo()
        {
            return m_redoStates.Any();
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach (ProjectState undoState in m_undoStates)
            {
                sb.AppendLine(undoState.Action);
            }
            sb.AppendLine("^^ current state ^^");
            foreach (ProjectState redoState in m_redoStates)
            {
                sb.AppendLine(redoState.Action);
            }
            return sb.ToString();
        }
    }
}
