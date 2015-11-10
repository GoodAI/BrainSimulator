using GoodAI.Core.Utils;
using System.Collections.Generic;

namespace GoodAI.Modules.Harm
{

    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Each stochastic return predictor has own DS, which contains subset of actions/vars
    /// of the Root Decision Space
    /// </summary>
    public interface IDecisionSpace
    {
        void AddVariable(int ind);
        void RemoveVariable(int ind);

        void AddAction(int ind);
        void RemoveAction(int ind);

        /// <summary>
        /// Read the current state while ignoring non-included variables
        /// </summary>
        /// <returns>list of indexes of maxVariables which contains only values of included variables</returns>
        int[] GetCurrentState();

        int GetLastExecutedAction();

        /// <summary>
        /// Return list of all actions that were just executed including abstract ones, not only those in the DS.
        /// </summary>
        /// <returns>List of aciton indexes</returns>
        List<int> GetLastExecutedActions();

        float GetReward();

        void PromoteUtilitiesToChilds(float[] utilities, float scale);

        bool IsActionIncluded(int no);
        bool IsVariableIncluded(int no);
    }


    class MyVariableLink
    {
        public bool[] ChildIncluded{ get; private set; }
        public float[] Weights { get; private set; }

        public MyVariableLink(int maxSize)
        {
            ChildIncluded = new bool[maxSize];
            Weights = new float[maxSize];
        }
    }

    public class MyActionLink
    {
        private List<bool> m_childIncluded;
        private List<float> m_weights;
        private MyRootDecisionSpace m_rds;

        public MyActionLink(int initSize, MyRootDecisionSpace rds)
        {
            this.m_rds = rds;
            m_childIncluded = new List<bool>(initSize);
            m_weights = new List<float>(initSize);
        }

        private void Resize(int index)
        {
            while (index >= m_childIncluded.Count)
            {
                m_childIncluded.Add(false);
                m_weights.Add(0);
            }
        }

        public bool IsIncluded(int index)
        {
            if (index >= m_rds.ActionManager.GetNoActions())
            {
                MyLog.ERROR.WriteLine("Action index out of range of max actions!");
                return false;
            }
            this.Resize(index);
            return m_childIncluded[index];
        }

        public float GetWeight(int index)
        {
            if (index >= m_rds.ActionManager.GetNoActions())
            {
                MyLog.ERROR.WriteLine("Action index out of range of max actions!");
                return 0;
            }
            this.Resize(index);
            return m_weights[index];
        }

        public void SetWeight(int index, float w)
        {
            if (index >= m_rds.ActionManager.GetNoActions())
            {
                MyLog.ERROR.WriteLine("Action index (" + index +
                    ") out of range of max actions! (" + m_rds.ActionManager.GetNoActions() + ")");
                return;
            }
            this.Resize(index);
            m_childIncluded[index] = true;
            m_weights[index] = w;
        }

        public void Remove(int index)
        {
            if (index >= m_rds.ActionManager.GetNoActions())
            {
                MyLog.ERROR.WriteLine("Action index out of range of max actions!");
                return;
            }
            this.Resize(index);
            m_childIncluded[index] = false;
            m_weights[index] = 0;
        }
    }

    public class MyDecisionSpace : IDecisionSpace
    {
        public static readonly float INIT_ACTION_WEIGHT = 1.0f;
        public static readonly float INIT_VAR_WEIGHT = 1.0f;

        private float m_prevValue = float.MinValue;
        private MyModuleParams m_setup;

        private MyRootDecisionSpace m_rds;
        private MyStochasticReturnPredictor m_mySRP;

        public MyActionLink ChildActions { get; private set; } 
        private MyVariableLink m_childVariables;

        /// <summary>
        /// Holds vector of indexes of used variables. If the variable is removed, its last
        /// value should be left in the vector (so that the Q(s,a) matrix coords do not move too much).
        /// </summary>
        private int[] m_S_t;


        public MyDecisionSpace(MyStochasticReturnPredictor mySRP, MyRootDecisionSpace rds, MyModuleParams setup)
        {
            this.m_setup = setup;
            this.m_rds = rds;
            this.m_mySRP = mySRP;
            m_S_t = new int[rds.VarManager.MAX_VARIABLES];

            ChildActions = new MyActionLink(0, rds);
            m_childVariables = new MyVariableLink(rds.VarManager.GetMaxVariables());
        }

        /// <summary>
        /// Determine one action that has been just executed. In case that multiple actions are found, 
        /// the one with the highest level is returned.
        /// </summary>
        /// <returns>Index of my action that has been executed, -1 if non of them.</returns>
        public int GetLastExecutedAction()
        {
            int executedInd = -1;
            List<MyMotivatedAction> actions = m_rds.ActionManager.Actions;

            for (int i = 0; i < m_rds.ActionManager.GetNoActions(); i++)
            {

                if (ChildActions.IsIncluded(i) && actions[i].JustExecuted())
                {
                    if (executedInd == -1)
                    {
                        executedInd = i;
                    }
                    else if (actions[i].GetLevel() > actions[executedInd].GetLevel())
                    {
                        int navA = m_rds.ActionManager.GetNoPrimitiveActinos() + 0;
                        int navB = m_rds.ActionManager.GetNoPrimitiveActinos() + 1;

                        // if this is navigation (on X or Y), do not set as executed
                        // because it is executed almost each step and shields the rest of actions
                        if (!(!m_setup.EnableAbstractNavigation && (i == navA || i == navB)))
                        {
                            executedInd = i;
                        }
                    }
                }
            }
            return executedInd;
        }

        public List<int> GetLastExecutedActions()
        {
            List<int> ac = new List<int>();
            List<MyMotivatedAction> actions = m_rds.ActionManager.Actions;

            for (int i = 0; i < m_rds.ActionManager.GetNoActions(); i++)
            {
                if (actions[i].JustExecuted())
                {
                    ac.Add(i);
                }
            }
            return ac;
        }

        public int[] GetCurrentState()
        {
            int[] rds_state = m_rds.VarManager.GetCurrentState();

            for (int i = 0; i < m_S_t.Length; i++)
            {
                if (m_childVariables.ChildIncluded[i])
                {
                    m_S_t[i] = rds_state[i];
                }
            }
            return m_S_t;
        }

        public bool IsActionIncluded(int no)
        {
            return ChildActions.IsIncluded(no);
        }

        public bool IsVariableIncluded(int no)
        {
            if (no < 0 || no >= m_childVariables.ChildIncluded.Length)
            {
                MyLog.DEBUG.WriteLine("child index out of range!");
                return false;
            }
            return m_childVariables.ChildIncluded[no];
        }

        private float MyCurrentPromotedVal()
        {
            if (m_mySRP.GetPromotedVariableIndex() < 0)
            {
                return 0;
            }
            if (m_rds.VarManager.GetVarNo(m_mySRP.GetPromotedVariableIndex()) == null)
            {
                return 0;
            }
            return m_rds.VarManager.GetVarNo(m_mySRP.GetPromotedVariableIndex()).Current;
        }

        public float GetReward()
        {
            if (m_prevValue == float.MinValue)
            {
                m_prevValue = this.MyCurrentPromotedVal();
                return 0;
            }
            if (m_prevValue != this.MyCurrentPromotedVal())
            {
                this.m_prevValue = this.MyCurrentPromotedVal();
                return m_setup.RewardScale;
            }
            return 0;
        }

        /// <summary>
        /// Adds action, if the action is abstract, adds also its promoted variable to the DS.
        /// </summary>
        /// <param name="index">Index of action to be added.</param>
        public void AddAction(int index)
        {
            ChildActions.SetWeight(index, INIT_ACTION_WEIGHT);

            // TODO allocates all actions in the memory (until the index of this given action)
            while (m_mySRP.Mem.GetNoActions() <= index)
            {
                m_mySRP.Mem.AddAction();
            }

            if (m_rds.ActionManager.Actions[index].GetLevel() > 0)
            {
                int promoted = m_rds.ActionManager.Actions[index].GetPromotedVariableIndex();
                this.AddVariable(promoted);
            }
        }

        /// <summary>
        /// The same in the opposite direction; assumed: since the action, which controls this
        /// variable is not needed, also the variable is not needed.
        /// </summary>
        /// <param name="index">Index of action to be removed</param>
        public void RemoveAction(int index)
        {
            ChildActions.Remove(index);
            m_mySRP.Mem.DisableAction(index);

            if (m_rds.ActionManager.Actions[index].GetLevel() > 0)
            {
                int promoted = m_rds.ActionManager.Actions[index].GetPromotedVariableIndex();
                this.RemoveVariable(promoted);
            }
        }

        public void AddVariable(int index)
        {
            if (index >= m_rds.VarManager.GetMaxVariables())
            {
                if (m_rds.VarManager.GetMaxVariables() > 0)
                {
                    MyLog.ERROR.WriteLine("Variable index (" + index +
                        ") out of range of max variables! (" + m_rds.VarManager.GetMaxVariables() + ")");
                }
                return;
            }
            m_childVariables.ChildIncluded[index] = true;
            m_childVariables.Weights[index] = INIT_VAR_WEIGHT;
        }

        public void RemoveVariable(int index)
        {
            if (index >= m_rds.VarManager.GetMaxVariables())
            {
                MyLog.ERROR.WriteLine("Variable index out of range of max variables!");
                return;
            }
            m_childVariables.ChildIncluded[index] = false;
            m_childVariables.Weights[index] = 0;
        }

        public void PromoteUtilitiesToChilds(float[] utils, float scale)
        {
            int pos = 0;
            for (int i = 0; i < m_rds.ActionManager.GetNoActions(); i++)
            {
                if (ChildActions.IsIncluded(i))
                {
                    m_rds.ActionManager.Actions[i].AddToMotivation(utils[pos] * scale);
                    pos++;
                }
            }
        }
    }
}
