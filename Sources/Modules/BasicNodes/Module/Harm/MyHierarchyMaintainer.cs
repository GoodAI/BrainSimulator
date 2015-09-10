using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Modules.Harm
{
    public interface ISubspacer
    {
        void SubspaceFor(int variableIndex);
    }

    /// <summary>
    /// Handles the correct: hierarchical decision-making, learning, hierarchy update
    /// </summary>
    public interface IHierarchyMaintainer : ISubspacer
    {
        /// <summary>
        /// Each SRP check own reward, executes own learning.
        /// In the current verison the hierarchical depedencies are ignored (because they can be). 
        /// </summary>
        void Learn();

        /// <summary>
        /// Infers utilities of primitive (all in fact) actions from the aciton hierarchy.
        /// The computed utilities are the final decision of the system.
        /// </summary>
        float[] InferUtilities();

        /// <summary>
        /// Checks conditions for subspacing, subspaces based on data in the action/variable traces.
        /// </summary>
        void Subspacing();

        /// <summary>
        /// Create new abstract action - one StochasticReturnPredictor which promotes one variable.
        /// Level of the abstraction is detedmined by the actions contained in the action space.
        /// Note that the level of abstraction is defined by the child ACTIONS here!
        /// </summary>
        /// <param name="variables">list of variables to place in the DS</param>
        /// <param name="actions">list of actions to the DS</param>
        /// <param name="promotedVariable">variable that is promoted by this SRP</param>
        /// <param name="label">name of this "action"</param>
        /// <returns>newly created abstract action</returns>
        MyStochasticReturnPredictor ManualSubspacing(int[] variables, int[] actions, int promotedVariable, String label);

        int GetNoPrimitiveActions();
    }

    public abstract class MyAbstractHierarchy : IHierarchyMaintainer
    {
        protected int m_step = 0;

        // for each level, there is a list of actions on the level
        protected List<List<MyAction>> m_actionLevels;

        // each variable (something that changed) should have own abstract action
        protected IStochasticReturnPredictor[] m_abstractActions;

        protected MyRootDecisionSpace m_rds;
        protected MyModuleParams m_learningParams;

        public MyAbstractHierarchy(MyRootDecisionSpace rds, MyModuleParams learningParams)
        {
            this.m_learningParams = learningParams;
            m_actionLevels = new List<List<MyAction>>();

            this.m_rds = rds;
            for (int i = 0; i < rds.ActionManager.GetNoActions(); i++)
            {
                this.AddAction(i);
            }
        }

        public abstract void NewVariableAdding();
        public abstract void Subspacing();

        public void AddAction(int index)
        {
            int level = m_rds.ActionManager.Actions[index].GetLevel();
            while (m_actionLevels.Count <= level)
            {
                m_actionLevels.Add(new List<MyAction>());
            }
            m_actionLevels[level].Add(m_rds.ActionManager.Actions[index]);
        }

        public int GetNoPrimitiveActions()
        {
            if (m_actionLevels == null || m_actionLevels[0] == null)
            {
                return 0;
            }
            return m_actionLevels[0].Count;
        }

        public void Learn()
        {
            // learn from the bottom of the hierarchy
            for (int i = 0; i < m_actionLevels.Count; i++)
            {
                for (int j = 0; j < m_actionLevels[i].Count; j++)
                {
                    MyAction a = m_actionLevels[i][j];

                    if (i > 0)
                    {
                        if (!(a is MyStochasticReturnPredictor))
                        {
                            MyLog.ERROR.WriteLine("Error, primitive aciton found on level > 0");
                            continue;
                        }
                        //MyLog.DEBUG.WriteLine("Learning for action: " + a.getLabel());
                        ((MyStochasticReturnPredictor)a).Learn();
                    }
                }
            }
        }

        public float[] InferUtilities()
        {
            // infer actions from the top of the hierarchy
            for (int i = m_actionLevels.Count - 1; i >= 0; i--)
            {
                for (int j = 0; j < m_actionLevels[i].Count; j++)
                {
                    MyAction a = m_actionLevels[i][j];

                    if (i > 0)
                    {
                        if (!(a is MyStochasticReturnPredictor))
                        {
                            MyLog.ERROR.WriteLine("Error, primitive aciton found on level > 0");
                            continue;
                        }
                        //MyLog.DEBUG.WriteLine("ASM for action: "+a.getLabel());
                        ((MyStochasticReturnPredictor)a).SelectAction();
                    }
                }
            }
            // list of utilities of primitive actions
            float[] utils = new float[m_actionLevels[0].Count];
            for (int i = 0; i < utils.Length; i++)
            {
                utils[i] = m_actionLevels[0][i].GetMyTotalMotivation();
            }

            return utils;
        }

        public void PostSimulationStep()
        {
            // order should not matter here
            for (int i = m_actionLevels.Count - 1; i >= 0; i--)
            {
                for (int j = 0; j < m_actionLevels[i].Count; j++)
                {
                    MyAction a = m_actionLevels[i][j];
                    a.PostSimulationStep();
                }
            }
            m_rds.VarManager.ShouldBeSubspaced.Clear();
            m_step++;
        }

        public List<List<MyAction>> GetActionLevels() { return this.m_actionLevels; }

        public abstract void SubspaceFor(int index);

        public abstract MyStochasticReturnPredictor
            ManualSubspacing(int[] variables, int[] actions, int promotedVariable, String label);

    }

    public abstract class MyAbsHierarchyMaintainer : MyAbstractHierarchy
    {
        public MyAbsHierarchyMaintainer(MyRootDecisionSpace rds, MyModuleParams learningParams)
            : base(rds, learningParams)
        {
        }
        
        public override void NewVariableAdding()
        {
            // all newly discovered variables..
            for (int i = 0; i < m_rds.VarManager.ShouldBeSubspaced.Count; i++)
            {
                // for all abstract actions
                for (int l = 1; l < this.m_actionLevels.Count; l++)
                {
                    for (int ac = 0; ac < this.m_actionLevels[l].Count; ac++)
                    {
                        if (this.m_actionLevels[l][ac] is MyStochasticReturnPredictor)
                        {
                            // add this new variable to the DS
                            ((MyStochasticReturnPredictor)this.m_actionLevels[l][ac]).AddNewVariable(m_rds.VarManager.ShouldBeSubspaced[i]);
                        }
                        else
                        {
                            MyLog.ERROR.WriteLine("Abstract action of type another than MySRP found");
                        }
                    }
                }
            }
        }

        public override void Subspacing()
        {
            if (m_step > m_learningParams.ActionLength && m_step > m_learningParams.VarLength)
            {
                for (int i = 0; i < m_rds.VarManager.ShouldBeSubspaced.Count; i++)
                {
                    if (m_rds.VarManager.GetVarNo(m_rds.VarManager.ShouldBeSubspaced[i]).MyAction == null)
                    {
                        this.SubspaceFor(m_rds.VarManager.ShouldBeSubspaced[i]);
                    }
                }
            }
        }

        public override void SubspaceFor(int index)
        {
            int pos = 0;
            int[] decodedActions;
            int[] decodedVariables;

            if (m_learningParams.SubspaceActions)
            {
                decodedActions = this.ChooseActions();
            }
            else
            {
                decodedActions = m_rds.ActionManager.GetPrimitiveActionIndexes();
            }

            if (m_learningParams.SubspaceVariables)
            {
                decodedVariables = this.ChooseVariables();
            }
            else
            {
                decodedVariables = new int[m_rds.VarManager.GetMaxVariables() - 1];
                pos = 0;
                for (int i = 0; i < m_rds.VarManager.GetMaxVariables(); i++)
                {
                    if (i != index)
                    {
                        decodedVariables[pos++] = i;
                    }
                }
            }

            ManualSubspacing(decodedVariables, decodedActions, index, m_rds.VarManager.GetVarNo(index).GetLabel());
        }

        protected abstract int[] ChooseActions();

        protected abstract int[] ChooseVariables();


        public override MyStochasticReturnPredictor ManualSubspacing(
        int[] variables, int[] actions, int promotedVariable, String label)
        {
            List<int> acs = actions.ToList();

            int maxChildActionLevel = 0;
            if (actions.Length > m_rds.ActionManager.GetNoActions())
            {
                MyLog.ERROR.WriteLine("Too many actions to be added, ignoring this DS!!");
                return null;
            }

            MyLog.INFO.Write("Hierarchy: action named " + label + " added, variables: ");
            MyStochasticReturnPredictor a = new MyStochasticReturnPredictor(m_rds, promotedVariable, m_learningParams, label, 0);
            for (int i = 0; i < variables.Length; i++)
            {
                if (variables[i] == promotedVariable)
                {
                    //MyLog.ERROR.WriteLine("Cannot add promoted variable into the DS, ignoring this one");
                    continue;
                }
                a.Ds.AddVariable(variables[i]);

                if (m_learningParams.BuildMultilevelHierarchy)
                {
                    if (m_rds.VarManager.GetVarNo(variables[i]).MyAction != null)
                    {
                        acs.Add(m_rds.VarManager.GetVarNo(variables[i]).MyAction.GetMyIndex());
                    }
                }
                MyLog.INFO.Write(" "+variables[i]);
            }
            MyLog.INFO.Write("\t actions: ");
            for (int i = 0; i < acs.Count; i++)
            {
                a.Ds.AddAction(acs[i]);
                MyLog.INFO.Write(" " + m_rds.ActionManager.GetActionLabel(acs[i]));

                if (m_rds.ActionManager.Actions[acs[i]].GetLevel() > maxChildActionLevel)
                {
                    maxChildActionLevel = m_rds.ActionManager.Actions[acs[i]].GetLevel();
                }
            }
            MyLog.INFO.WriteLine();
            // level is determined by the most abstract child
            a.SetLevel(maxChildActionLevel + 1);
            m_rds.ActionManager.AddAction(a);
            this.AddAction(m_rds.ActionManager.GetNoActions() - 1);
            return a;
        }

        private int GetMaxChildLevel()
        {
            return -1;
        }
    }

    public class MyHierarchyMaintainer : MyAbsHierarchyMaintainer
    {
        public MyHierarchyMaintainer(MyRootDecisionSpace rds, MyModuleParams learningParams)
            : base(rds, learningParams)
        {
        }

        protected override int[] ChooseActions()
        {
            IActionHistory ah = m_rds.ActionManager.AcitonHistory;

            Dictionary<int, float> importances = new Dictionary<int,float>();
            int action;
            float w;

            for (int i = 0; i < m_learningParams.ActionLength; i++)
            {
                action = ah.GetActionExecutedBefore(i);
                if (action != -1)
                {
                    w = 1.0f - ((float)i / (float)m_learningParams.ActionLength);
                    
                    if (importances.ContainsKey(action))
                    {
                        importances[action] += w;
                    }
                    else
                    {
                        importances.Add(action, w);
                    }
                }
            }
            
            List<int> chosen = new List<int>();
            foreach (var item in importances)
            {
                if (item.Value >= m_learningParams.ActionSubspacingThreshold)
                {
                    chosen.Add(item.Key);
                }
            }
            return chosen.ToArray();
        }

        protected override int[] ChooseVariables()
        {
            IVariableHistory vh = m_rds.VarManager.VarHistory;
            Dictionary<int, float> importances = new Dictionary<int, float>();
            List<int> var;
            float w;

            for (int i = 0; i < m_learningParams.VarLength; i++)
            {
                var = vh.GetVariablesChangedBefore(i);
                if (var != null)
                {
                    // all variables can change at one step
                    w = (1.0f - ((float)i / (float)m_learningParams.VarLength)) / (float)m_learningParams.VarLength;

                    for (int j = 0; j < var.Count; j++)
                        if (importances.ContainsKey(var[j]))
                        {
                            importances[var[j]] += w;
                        }
                        else
                        {
                            importances.Add(var[j], w);
                        }
                }
            }
            List<int> chosen = new List<int>();
            foreach (var item in importances)
            {
                MyLog.DEBUG.WriteLine("dictionary contains: " + item.Key + ": " + item.Value);
                if (item.Value >= m_learningParams.VariableSubspacingThreshold)
                {
                    chosen.Add(item.Key);
                }
            }
            return chosen.ToArray();
        }
    }
}
