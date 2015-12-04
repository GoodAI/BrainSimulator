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
        public string SerializedProject;
    }

    public class UndoManager
    {

        public uint HistorySize { get; private set; }

        // TODO: Change state from string to a structure (the selected node will be also stored etc.)

        // The history. m_undoStates.First is the oldest item in the history.
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

            while (m_undoStates.Count >= HistorySize)
                m_undoStates.RemoveFirst();

            m_undoStates.AddLast(state);
        }

        public ProjectState Undo()
        {
            if (!m_undoStates.Any())
                return null;

            ProjectState lastState = m_undoStates.Last();

            m_undoStates.RemoveLast();

            m_redoStates.Push(lastState);

            return lastState;
        }

        public ProjectState Redo()
        {
            if (!m_redoStates.Any())
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

        public void Reset(ProjectState firstState)
        {
            Clear();
            SaveState(firstState);
        }
    }
}
