using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace GoodAI.Modules.School.LearningTasks.TransducerTasks
{
    public class FiniteTransducerTransition
    {
        public int from;
        public int to;
        public int symbol;
        public int action;

        public FiniteTransducerTransition(int from, int to, int symbol, int action)
        {
            this.from = from;
            this.to = to;
            this.symbol = symbol;
            this.action = action;
        }
    }

    public class FiniteTransducer
    {
        // setup
        protected int NumOfStates { get; set; }
        protected int NumOfSymbols { get; set; }
        protected int NumOfActions { get; set; }

        protected FiniteTransducerTransition[,,] m_transitionTable;
        protected int m_initialState = 0;
        protected HashSet<int> m_finalStates;

        // runtime
        protected int m_currentState;
        protected Random m_rnd;

        public FiniteTransducer(int numOfStates, int numOfSymbols, int numOfActions)
        {
            NumOfStates = numOfStates;
            NumOfSymbols = numOfSymbols;
            NumOfActions = numOfActions;

            m_transitionTable = new FiniteTransducerTransition[numOfStates, numOfStates, numOfSymbols];
            m_finalStates = new HashSet<int>();

            m_currentState = 0;
            m_rnd = new System.Random();
        }

        public void AddTransition(int from, int to, int symbol, int action)
        {
            AssertRange(from, 0, NumOfStates - 1);
            AssertRange(to, 0, NumOfStates - 1);
            AssertRange(symbol, 0, NumOfSymbols - 1);
            AssertRange(action, 0, NumOfActions - 1);

            m_transitionTable[from, to, symbol] = new FiniteTransducerTransition(from, to, symbol, action);
        }

        public void SetInitialState(int state)
        {
            AssertRange(state, 0, NumOfStates - 1);

            m_initialState = state;
        }

        public void AddFinalState(int state)
        {
            AssertRange(state, 0, NumOfStates - 1);

            m_finalStates.Add(state);
        }

        public void Start()
        {
            m_currentState = m_initialState;
        }

        public void UseTransition(FiniteTransducerTransition t)
        {
            Debug.Assert(t.from == m_currentState);
            m_currentState = t.to;
        }

        public FiniteTransducerTransition pickNextTransitionRandomly()
        {
            FiniteTransducerTransition picked = null;

            List<FiniteTransducerTransition> options = new List<FiniteTransducerTransition>(NumOfStates); // some default capacity
            for (int i = 0; i < NumOfStates; i++)
            {
                for (int j = 0; j < NumOfSymbols; j++)
                {
                    if (m_transitionTable[m_currentState, i, j] != null)
                        options.Add(m_transitionTable[m_currentState, i, j]);
                }
            }
            if (options.Count > 0)
            {
                picked = options[m_rnd.Next(0, options.Count)];
            }

            return picked;
        }

        void AssertRange(int value, int min, int max)
        {
            Debug.Assert(value >= min && value <= max);
        }
    }
}