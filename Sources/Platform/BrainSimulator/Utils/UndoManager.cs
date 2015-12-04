using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;

namespace GoodAI.BrainSimulator.Utils
{
    public class ProjectState
    {
        public string SerializedProject { get; private set; }
        public string ProjectPath { get; set; }

        public ProjectState(string serializedProject)
        {
            if (serializedProject == null)
                throw new ArgumentNullException("serializedProject");

            SerializedProject = serializedProject;
        }

        public override bool Equals(object obj)
        {
            return GetHashCode().Equals(obj.GetHashCode());
        }

        public override int GetHashCode()
        {
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

        public uint HistorySize { get; private set; }

        // TODO: Change state from string to a structure (the selected node will be also stored etc.)

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
            return m_undoStates.Count > 1;
        }

        public bool CanRedo()
        {
            return m_redoStates.Any();
        }
    }
}
