using System;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Modules.Harm
{
    public interface IActionAdder
    {
        void PerformActionAdding(List<int> actionsJustExecuted);
    }

    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// SRP is a standalone system, which includes: decision space, learning algorithm, action selection.
    /// </summary>
    public interface IStochasticReturnPredictor
    {
        /// <summary>
        /// Update the Q(s,a) memory
        /// </summary>
        void Learn();

        /// <summary>
        /// Compute Utilities, Select action (changes utility values), scale utilities, propagate them to childs.
        /// </summary>
        float[] SelectAction();

        /// <summary>
        /// Should be called after the simulation step.
        /// </summary>
        void PostSimulationStep();

        /// <summary>
        /// for scaling memory values to the interval 0,1
        /// </summary>
        /// <returns></returns>
        float GetMaxMemoryValue();

        /// <summary>
        /// If new variable is detected in the DS, it should be added into all decision spaces,
        /// knowledge then can be shared from the old memory. 
        /// 
        /// Then the strategy is updated for nev values by learning. 
        /// If both strategies (old and new one) are "the same", the variable can be then removed.
        /// </summary>
        /// <param name="no">index of new variale</param>
        void AddNewVariable(int no);
    }

    /// <summary>
    /// Is an abstract action, which lears in own decision space to get own type of reward.
    /// </summary>
    public class MyStochasticReturnPredictor : MyMotivatedAction, IStochasticReturnPredictor, IActionAdder
    {
        public MyRootDecisionSpace Rds {get; set; }
        public IDecisionSpace Ds { get; set; }
        public MyQSAMemory Mem { get; set; }
        public MyDiscreteQLearning LearningAlgorithm { get; set; }
        private MyMotivationBasedDeleteUnselectedASM m_asm;
        private MyLocalVariableHistory m_mlvh;

        private int m_prevSelectedAction;

        private int[] m_prev_st;  // previous state (for variable adding and sharing knowledge)
        private List<int> m_newVariables;

        public MyStochasticReturnPredictor(MyRootDecisionSpace rds, int myPromotedVariable,
            MyModuleParams setup, String label, int level)
            : base(label, level, setup)
        {
            base.AddPromotedVariable(myPromotedVariable, rds);

            this.Rds = rds;
            Ds = new MyDecisionSpace(this, rds, setup);
            Mem = new MyQSAMemory(rds.VarManager.MAX_VARIABLES, 0);
            m_asm = new MyMotivationBasedDeleteUnselectedASM(setup);
            LearningAlgorithm = new MyDiscreteQLearning(setup, Mem);
            m_mlvh = new MyLocalVariableHistory(rds, m_setup, Ds);
            m_prevSelectedAction = 0;
            m_prev_st = Ds.GetCurrentState();
            this.m_newVariables = new List<int>();
        }

        public void Learn()
        {
            int[] s_tt = Ds.GetCurrentState();
            float r_t = Ds.GetReward();
            int a_t = Ds.GetLastExecutedAction();

            if (a_t == this.m_prevSelectedAction)
            {
                base.SetJustExecuted(true);
            }

            if (a_t < 0)
            {
                // aciton ignored and nothing happened
                if (r_t == 0)
                {
                    return;
                }
                // action adding, reward received as a result of unknown action
                List<int> toBeAdded = Ds.GetLastExecutedActions();
                this.PerformActionAdding(toBeAdded);
                return;
            }

            if (m_setup.OnlineSubspaceVariables)
            {
                m_mlvh.monitorVariableChanges(this);
            }

            LearningAlgorithm.Learn(r_t, s_tt, a_t);

            if (r_t > 0)
            {
                // SRP is also an action, if the reward is received means that it just has been executed
                base.SetJustExecuted(true);
                this.ResetMotivationSource();
                m_mlvh.performOnlineVariableRemoving(this);
            }
        }

        public void AddNewVariable(int no)
        {
            // if the variable subspacing is disabled, all constants are in this DS too
            if(!Ds.IsVariableIncluded(no))
            {
                Ds.AddVariable(no);
                this.m_newVariables.Add(no);
            }
        }

        /// <summary>
        /// This adds all primitive actions, that were just executed. 
        /// Note that multilevel hierarchy is built by variable adding in the current version.
        /// </summary>
        /// <param name="actionsJustExecuted">lsit of actions that were just executed</param>
        public void PerformActionAdding(List<int> actionsJustExecuted)
        {
            for (int i = 0; i < actionsJustExecuted.Count; i++)
            {
                if (Rds.ActionManager.Actions[actionsJustExecuted[i]].GetLevel() == 0)
                {
                    this.Ds.AddAction(actionsJustExecuted[i]);
                }
            }
        }

        public float GetMaxMemoryValue()
        {
            return LearningAlgorithm.GetMaxVal();
        }

        public float[] SelectAction()
        {
            int[] s_tt = Ds.GetCurrentState();
            float[] utils;

            // use own asm to select one action?
            if (m_setup.UseHierarchicalASM)
            {
                utils = this.Rescale(m_asm.SelectAction(Mem.ReadData(s_tt)));
            }
            else
            {
                utils = this.Rescale(Mem.ReadData(s_tt));
            }
            if (m_setup.PropagateUtilitiesInHierarchy)
            {
                float myMot = this.GetMyTotalMotivation();
                Ds.PromoteUtilitiesToChilds(utils, myMot);
            }
            this.MarkSelection(utils);
            return utils;
        }

        private float[] Rescale(float[] utils)
        {
            if (this.GetMaxMemoryValue() == 0)
            {
                return utils;
            }
            for (int i = 0; i < utils.Length; i++)
            {
                utils[i] = utils[i] / this.GetMaxMemoryValue();
            }
            return utils;
        }

        private void MarkSelection(float[] utils)
        {
            float maxValue = utils.Max();
            this.m_prevSelectedAction = utils.ToList().IndexOf(maxValue);
        }

        public override void PostSimulationStep()
        {
            // prepare for new inference of motivaiton from the hierarchy
            base.PostSimulationStep();
            base.m_motivatonSource.MakeStep();

            m_prev_st = Ds.GetCurrentState();
            this.m_newVariables.Clear();
        }
    }
}
