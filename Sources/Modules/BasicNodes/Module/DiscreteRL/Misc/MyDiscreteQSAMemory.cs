using GoodAI.Core.Utils;
using System.Collections.Generic;

namespace GoodAI.Modules.Harm
{
    /// <summary>
    /// Defines the implementation of memory with adaptable dimensions and dimension sizes. 
    /// </summary>
    public interface IDiscreteQSAMemory
    {
        /// <summary>
        /// Place the utility of action into the memory, memory allocated if needed.
        /// </summary>
        /// <param name="stateValues">Array of indexes definind the current state</param>
        /// <param name="action">Current action to be written in the state</param>
        /// <param name="value">Utility value of the action</param>
        void WriteData(int[] stateValues, int action, float value);

        /// <summary>
        /// Set the dim. list sizes to 0 in the simplest case. This indicates that the dimension is not used.
        /// 
        /// Possible improvement1: recursivelly deallocate the memory
        /// Possible improvement2: average all utilities accross the deleted dimension (knowledge reuse).
        /// </summary>
        /// <param name="no">index of dimension to be deleted</param>
        void DisableDimension(int no);

        void DisableAction(int no);

        void AddAction();

        /// <summary>
        /// Read utility values in a given state. 
        /// If there is no data for action, the default values are returned.
        /// </summary>
        /// <param name="states">Array of indexes describing the curren state</param>
        /// <returns>Array of aciton utilities, for each of currently used (and non-disabled) actions one value</returns>
        float[] ReadData(int[] states);

        /// <summary>
        /// Get list of sizes of particular dimensions (the last one is (max) no. of actions)
        /// </summary>
        /// <returns>List of dimension sizes</returns>
        int[] GetDimensionSizes();

        /// <summary>
        /// Array of indexes used for indexing the matrix is finite -> also the max no. of state variables.
        /// </summary>
        /// <returns>Returns max number of state variables</returns>
        int GetMaxStateVariables();
    }

    public class MyQSAMemory : IDiscreteQSAMemory
    {
        private readonly int m_maxDimensions;
        private readonly int ACTIONS;

        // list of sizes of all dimensions (no actions is the last one)
        private int[] m_dimensionSizes;
        private Dictionary<int, bool> m_disabledActions;
        private MyDimList m_d;

        public MyQSAMemory(int maxStateVariables, int initNoActions)
        {
            m_maxDimensions = maxStateVariables + 1;
            ACTIONS = m_maxDimensions - 1;

            m_dimensionSizes = new int[m_maxDimensions];
            m_disabledActions = new Dictionary<int, bool>();

            m_d = new MyDimList(m_maxDimensions, 0);
            MyDimList tmp = m_d;

            // recursively init all dimensions
            for (int i = 1; i < m_maxDimensions; i++)
            {
                m_dimensionSizes[i] = 1;  // all constants by default

                tmp = tmp.GetChildNo(0);
                tmp = new MyDimList(m_maxDimensions, i);
            }

            this.m_dimensionSizes[ACTIONS] = initNoActions;
        }

        public int GetMaxStateVariables()
        {
            return this.m_maxDimensions - 1;
        }

        // If there is insufficient no of acitons, resize the memory
        public void CheckNoActions(int a_t)
        {
            while (m_dimensionSizes[ACTIONS] <= a_t)
            {
                m_dimensionSizes[ACTIONS]++;
            }
        }

        public void DisableDimension(int no)
        {
            if (no >= this.m_maxDimensions - 1)
            {
                MyLog.ERROR.WriteLine("Index of disabled dimension out of range!");
                return;
            }
            m_dimensionSizes[no] = 0;
        }

        public void DisableAction(int no)
        {
            if (no >= m_dimensionSizes[ACTIONS])
            {
                MyLog.ERROR.WriteLine("This action has not been used, ignoring");
                return;
            }
            if (!m_disabledActions.ContainsKey(no))
            {
                m_disabledActions.Add(no, true);
            }
        }

        public void AddAction()
        {
            m_dimensionSizes[ACTIONS]++;
        }

        public int GetNoActions()
        {
            return m_dimensionSizes[ACTIONS];
        }

        public void WriteData(int[] stateValues, int action, float value)
        {
            if (m_disabledActions.ContainsKey(action))
            {
                MyLog.ERROR.WriteLine("This action is disabled, will not set it!");
                return;
            }
            this.UpdateDimensionSizes(stateValues, action);
            RecursiveSet(stateValues, action, 0, m_d, value);
        }

        private void RecursiveSet(int[] stateValues, int action, int depth, MyDimList d, float value)
        {
            if (depth == this.m_maxDimensions - 2)
            {
                MyDimList last = d.GetChildNo(action);
                last.SetValue(value);
                return;
            }
            MyDimList child = d.GetChildNo(stateValues[depth]);
            this.RecursiveSet(stateValues, action, depth+1, child, value);
        }

        public float[] ReadData(int[] states)
        {
            this.UpdateDimensionSizes(states);
            float[] actionUtils = this.RecursiveRead(states, 0, m_d);
            return actionUtils;
        }

        private float[] RecursiveRead(int[] indexes, int depth, MyDimList d)
        {
            if (depth == this.m_maxDimensions - 2)
            {
                int noActions = m_dimensionSizes[ACTIONS] - m_disabledActions.Count;
                float[] utils = new float[noActions];

                // fill the utils array with the utilities of actions that are not disabled
                int pos = 0;

                for (int i = 0; i < m_dimensionSizes[ACTIONS]; i++)
                {
                    MyDimList tmp = d.GetChildNo(i);

                    if (!d.GetChildNo(i).IsLast())
                    {
                        MyLog.ERROR.WriteLine("Expexcted last dimension!");
                        utils[i] = MyDimList.DEF_VAL;
                    }
                    if (!m_disabledActions.ContainsKey(i))
                    {
                        utils[pos++] = d.GetChildNo(i).GetValue();
                    }
                }
                return utils;
            }
            else
            {
                MyDimList child = d.GetChildNo(indexes[depth]);
                return RecursiveRead(indexes, depth + 1, child);
            }
        }

        public int[] GetGlobalActionIndexes()
        {
            int[] indexes = new int[m_dimensionSizes[ACTIONS] - m_disabledActions.Count];
            int pos = 0;
            for (int i = 0; i < m_dimensionSizes[ACTIONS]; i++)
            {
                if(!m_disabledActions.ContainsKey(i))
                {
                    indexes[pos++] = i;
                }
            }
            return indexes;
        }

        public int[] GetStateSizes()
        {
            int[] sizes = new int[m_dimensionSizes.Length - 1];
            for (int i = 0; i < sizes.Length; i++)
            {
                sizes[i] = m_dimensionSizes[i];
            }
            return sizes;
        }

        public int[] GetDimensionSizes()
        {
            return m_dimensionSizes;
        }

        private void UpdateDimensionSizes(int[] stateIndexes, int action)
        {
            this.UpdateDimensionSizes(stateIndexes);
            if (m_dimensionSizes[ACTIONS] <= action)
            {
                m_dimensionSizes[ACTIONS] = action + 1;
            }
        }

        private void UpdateDimensionSizes(int[] stateIndexes)
        {
            if (stateIndexes.Length != m_maxDimensions - 1)
            {
                MyLog.ERROR.WriteLine("incorrect no of dimensions! Asked for: "+
                    stateIndexes.Length+" but dimensions are: " + (m_maxDimensions-1)+ " "+
                    m_dimensionSizes.Length);
                return;
            }
            for (int i = 0; i < stateIndexes.Length; i++)
            {
                if (m_dimensionSizes[i] <= stateIndexes[i])
                {
                    m_dimensionSizes[i] = stateIndexes[i] + 1;
                }
            }
        }
    }
}

