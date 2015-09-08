using System;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Modules.Harm
{
    /// <summary>
    /// Global config class
    /// </summary>
    public class MyModuleParams
    {
        public float Alpha;
        public float Lambda;
        public float Gamma;
        public int TraceLength = 1;
        public bool UseTrace;
        public float MotivationChange;
        public float MinEpsilon;
        public float RewardScale;
        public bool UseHierarchicalASM;
        public int VarLength;
        public int ActionLength;
        public float ActionSubspacingThreshold;
        public float VariableSubspacingThreshold;
        public float HistoryForgettingRate;
        public bool SubspaceActions;
        public bool SubspaceVariables;
        public bool BuildMultilevelHierarchy;
        public bool EnableAbstractNavigation;
        public float HierarchicalMotivationScale;
        public bool PropagateUtilitiesInHierarchy;

        public bool OnlineSubspaceVariables;
        public float OnlineHistoryForgettingRate;
        public float OnlineVariableRemovingThreshold;
    }

    /// <summary>
    /// Remembers history of past N steps, learning algorithm computes delta 
    /// (one-step difference between expected and observed state value) and spreads this
    /// discounted value back into the history. This speeds up learning significantly.
    /// </summary>
    public class MyEligibilityTrace
    {
        private List<TraceData> m_traceData;    // position 0 stores the most recent state-aciton
        private MyModuleParams m_learning;
        public float[] Lambda;                  // precomputed lambda^(t+1)
        
        public class TraceData
        {
            public readonly int[] S_t;
            public int A_t;

            public TraceData(int[] s_t, int a_t)
            {
                this.S_t = s_t;
                this.A_t = a_t;
            }
        }

        public MyEligibilityTrace(MyModuleParams learning)
        {
            this.m_learning = learning;
            m_traceData = new List<TraceData>();
            Lambda = new float[learning.TraceLength];
            float tmp = learning.Lambda;

            for (int i = 0; i < learning.TraceLength; i++)
            {
                Lambda[i] = tmp;
                tmp *= learning.Lambda;
            }
        }

        public void UpdateTraceValues()
        {
            // update the length of lambda vector (if changed from GUI)
            if (m_learning.TraceLength != Lambda.Length)
            {
                float tmp = m_learning.Lambda;
                Lambda = new float[m_learning.TraceLength];
                for (int i = 0; i < m_learning.TraceLength; i++)
                {
                    Lambda[i] = tmp;
                    tmp *= m_learning.Lambda;
                }
            }
            // update the size of trace if needed
            while (this.m_traceData.Count >= m_learning.TraceLength && m_traceData.Count > 0)
            {
                m_traceData.RemoveAt(m_traceData.Count - 1);
            }
        }

        public void PushState(int[] s_t)
        {
            if (m_traceData.Count > 0 && s_t.Length != m_traceData[0].S_t.Length)
            {
                m_traceData.Clear();
            }
            while (m_traceData.Count >= m_learning.TraceLength)
            {
                m_traceData.RemoveAt(m_traceData.Count - 1);
            }
            int[] inserted = s_t.Clone() as int[];
            TraceData tmp = new TraceData(inserted, 0);
            m_traceData.Insert(0, tmp);
        }

        public int Size()
        {
            return m_traceData.Count;
        }

        public TraceData Get(int i)
        {
            return m_traceData[i];
        }
    }

    public class MyDiscreteQLearning
    {
        private MyModuleParams m_learning;
        private MyQSAMemory m_mem;
        private MyEligibilityTrace m_trace;

        private float m_maxValue;

        public MyDiscreteQLearning(MyModuleParams learningParams, MyQSAMemory mem)
        {
            this.m_mem = mem;
            this.m_learning = learningParams;

            m_trace = new MyEligibilityTrace(learningParams);
            m_trace.PushState(new int[mem.GetMaxStateVariables()]);
        }

        public float GetMaxVal()
        {
            return this.m_maxValue;
        }

        private void UpdateMax(float val)
        {
            if (this.m_maxValue < val)
            {
                this.m_maxValue = val;
            }
        }

        /// <summary>
        /// Performs Q-Learning (including the Eligibility Trace if enabled).
        /// </summary>
        /// <param name="r_t">Reward received as a result of previous action</param>
        /// <param name="s_tt">Current state s_t'</param>
        /// <param name="a_t">Previously executed action</param>
        public void Learn(float r_t, int[] s_tt, int a_t)
        {
            m_trace.UpdateTraceValues();
            m_trace.Get(0).A_t = a_t;

            float Q_st_at;
            float result;

            // standard Q-learning..
            float[] utils = m_mem.ReadData(s_tt);       // utility values in the s_t'
            float max = utils.Max();

            m_mem.CheckNoActions(a_t);

            Q_st_at = m_mem.ReadData(m_trace.Get(0).S_t)[m_trace.Get(0).A_t];

            float delta = r_t + m_learning.Gamma * max - Q_st_at;
            result = Q_st_at + m_learning.Alpha * delta;

            this.UpdateMax(result);
            m_mem.WriteData(m_trace.Get(0).S_t, m_trace.Get(0).A_t, result);

            // eligibility, update Q(s,a) multiple steps back based on delta
            if (m_learning.UseTrace)
            {
                for (int i = 0; i < m_trace.Size(); i++)
                {
                    Q_st_at = m_mem.ReadData(m_trace.Get(i).S_t)[m_trace.Get(i).A_t];
                    result = Q_st_at + m_learning.Alpha * delta * m_trace.Lambda[i];
                    this.UpdateMax(result);
                    m_mem.WriteData(m_trace.Get(i).S_t, m_trace.Get(i).A_t, result);
                }
            }
            m_trace.PushState(s_tt);
        }
       
        public float[] ReadUtils(int[] s_t)
        {
            float[] tmp = m_mem.ReadData(s_t);
            return tmp;
        }
    }

    public interface IActionSelectionMethod
    {
        float[] SelectAction(float[] utilities);
    }

    /// <summary>
    /// Sets motivation values of non-selected acitons to the predefined value (e.g. 0).
    /// </summary>
    public class MyMotivationBasedDeleteUnselectedASM : IActionSelectionMethod
    {
        private MyModuleParams m_setup;
        private float m_motivation;
        private Random m_rnd;

        private float m_unselectedValue = 0;

        public MyMotivationBasedDeleteUnselectedASM(MyModuleParams setup)
        {
            this.m_setup = setup;
            m_rnd = new Random();
        }

        public void SetMotivatoin(float motivation)
        {
            this.m_motivation = motivation;
        }

        public void SetUnselectedValue(float unselected)
        {
            this.m_unselectedValue = unselected;
        }

        private int FindMaxInd(float[] util)
        {
            int maxInd = 0;
            bool multiple = false;

            for (int i = 0; i < util.Length; i++)
            {
                if (util[maxInd] < util[i])
                {
                    maxInd = i;
                    multiple = false;
                }
                else if (util[maxInd] == util[i])
                {
                    multiple = true;
                }
            }
            if (multiple)
                return -1;
            return maxInd;
        }

        private float[] DeleteAllExcept(float[] utils, int selected)
        {
            for (int i = 0; i < utils.Length; i++)
            {
                if (i == selected)
                {
                    continue;
                }
                utils[i] = this.m_unselectedValue;
            }
            return utils;
        }

        public float[] SelectAction(float[] actionUtilities)
        {
            float epsilon = 1 - m_motivation;
            int maxInd = this.FindMaxInd(actionUtilities);
            float[] output = actionUtilities.Clone() as float[];
            int selected = 0;

            if (maxInd < 0) // if multiple equal utilities found, select randomly
            {
                epsilon = 1;
            }
            else if (epsilon < m_setup.MinEpsilon)
            {
                epsilon = m_setup.MinEpsilon;
            }

            if (m_rnd.NextDouble() <= epsilon)
            {
                selected = m_rnd.Next(output.Length);
            }
            else
            {
                selected = maxInd;
            }
            return DeleteAllExcept(output, selected);
        }
    }

}
