using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;

namespace GoodAI.Modules.Harm
{
    /// <summary>
    /// Holds all known actions and variables.
    /// </summary>
    public class MyRootDecisionSpace
    {
        public ActionManager ActionManager{ get; set; }
        public VariableManager VarManager{ get; set; }
        
        public MyRootDecisionSpace(int maxVariables, String[] actionNames, MyModuleParams setup)
        {
            VarManager = new VariableManager(maxVariables, setup);
            ActionManager = new ActionManager(actionNames, setup);
        }
    }
    
    public abstract class MyAstractDecisionSpaceUnit
    {
        protected int m_level;
        protected String m_label;

        public MyAstractDecisionSpaceUnit(String label, int level)
        {
            this.m_level = level;
            this.m_label = label;
        }

        public virtual void SetLevel(int level)
        {
            if (level < 0)
            {
                MyLog.ERROR.WriteLine("Cannot set level under the 0");
                return;
            }
            this.m_level = level;
        }

        public int GetLevel()
        {
            return this.m_level;
        }

        public String GetLabel()
        {
            return this.m_label;
        }

        public void SetLabel(String label)
        {
            this.m_label = label;
        }
    }

    public class MyAction : MyAstractDecisionSpaceUnit
    {
        private bool m_executed;
        protected int m_myPromotedVariableInd;
        protected float m_motivationFromParents;

        public MyAction(String label, int level)
            : base(label, level)
        {
            this.m_motivationFromParents = 0;
            this.m_executed = false;
            this.m_myPromotedVariableInd = -1;
        }

        public override void SetLevel(int level)
        {
            base.SetLevel(level);
            if (level == 0)
            {
                m_myPromotedVariableInd = -1;
            }
        }

        public int GetPromotedVariableIndex()
        {
            return m_myPromotedVariableInd;
        }

        /// <summary>
        /// Works with only one variable
        /// </summary>
        /// <param name="varIndex"></param>
        public void AddPromotedVariable(int varIndex, MyRootDecisionSpace rds)
        {
            m_myPromotedVariableInd = varIndex;
            rds.VarManager.GetVarNo(varIndex).SetMyAction((MyStochasticReturnPredictor)this);
        }

        /// <summary>
        /// Primitive actions are updated on the bottom, abstract aciton sets this by itself.
        /// </summary>
        /// <param name="yes">Whether the action (or child action) has just been executed.</param>
        public void SetJustExecuted(bool yes)
        {
            this.m_executed = yes;
        }

        /// <summary>
        /// This indicates that the action has been executed in the current step. 
        /// It is used in the higher levels of hierarchy: higher level action "hides" execution
        /// of lower level actions that are part of its decision space (policy). 
        /// </summary>
        /// <returns>True if the action (or some acton from my decuison sapce) has jsut been executed.</returns>
        public bool JustExecuted()
        {
            return this.m_executed;
        }

        public void ClearMotivationFromParents()
        {
            this.m_motivationFromParents = 0;
        }

        public void AddToMotivation(float value)
        {
            this.m_motivationFromParents += value;
        }

        /// <summary>
        /// TotalUtility is composed of sum of utilities promoted from parents and my Motivation (0 here).
        /// </summary>
        /// <returns>Total utility of the action (motivation*RS + utils.from parents)</returns>
        public virtual float GetMyTotalMotivation()
        {
            return this.m_motivationFromParents;
        }

        public virtual void PostSimulationStep()
        {
            this.SetJustExecuted(false);
            this.ClearMotivationFromParents();
        }
    }

    /// <summary>
    /// Action, that has own source of motivation with own inner dynamics.
    /// </summary>
    public class MyMotivatedAction : MyAction
    {
        protected MyMotivationSource m_motivatonSource;
        protected MyModuleParams m_setup;
        private int m_myIndex = -1;

        public MyMotivatedAction(String label, int level, MyModuleParams setup)
            : base(label, level)
        {
            this.m_motivatonSource = new MyMotivationSource(setup, level);
            this.m_setup = setup;
        }

        public int GetMyIndex()
        {
            return this.m_myIndex;
        }

        public void SetMyIndex(int myIndex)
        {
            this.m_myIndex = myIndex;
        }

        public override float GetMyTotalMotivation()
        {
            if (base.m_motivationFromParents > this.m_motivatonSource.GetMotivation())
            {
                return base.m_motivationFromParents;
            }
            return this.m_motivatonSource.GetMotivation();
        }

        public void ResetMotivationSource()
        {
            this.m_motivatonSource.Reset();
        }

        public void OverrideMotivationSourceWith(float val)
        {
            this.m_motivatonSource.OverrideWith(val);
        }
    }

    /// <summary>
    /// Stores all aciton that the agent is capable of (primitive and abstract ones).
    /// New actions are added online, the old ones are disabled in the particular spaces.
    /// </summary>
    public class ActionManager
    {
        public List<MyMotivatedAction> Actions{ get; set; }
        public IActionHistory AcitonHistory{ get; set; }
        protected MyModuleParams m_setup;

        public ActionManager(int noPrimitiveActions, MyModuleParams setup)
        {
            this.m_setup = setup;
            Actions = new List<MyMotivatedAction>();
            for (int i = 0; i < noPrimitiveActions; i++)
            {
                Actions.Add(new MyMotivatedAction("" + i, 0, setup));
            }
            this.AcitonHistory = new MyActionHistory(this, setup);
            this.AcitonHistory.AddAllCurrentActions();
        }

        public ActionManager(String[] labels, MyModuleParams setup)
        {
            this.m_setup = setup;
            Actions = new List<MyMotivatedAction>();
            for (int i = 0; i < labels.Length; i++)
            {
                Actions.Add(new MyMotivatedAction(labels[i], 0, setup));
            }
            this.AcitonHistory = new MyActionHistory(this, setup);
            this.AcitonHistory.AddAllCurrentActions();
        }

        public int GetNoActions()
        {
            return Actions.Count;
        }

        public void AddAction(MyMotivatedAction a)
        {
            Actions.Add(a);
            a.SetMyIndex(Actions.Count - 1);
            AcitonHistory.AddAction(this.Actions.Count - 1);
        }

        public void SetJustExecuted(int action)
        {
            if (action < 0 || action >= this.GetNoActions())
            {
                MyLog.ERROR.WriteLine("Index out of range");
                return;
            }
            this.AcitonHistory.RegisterExecutedAction(action);
            this.Actions[action].SetJustExecuted(true);
        }

        public String[] GetActionLabels(int[] indexes)
        {
            String[] labels = new String[indexes.Length];
            for(int i=0; i<labels.Length; i++)
            {
                labels[i] = GetActionLabel(indexes[i]);
            }
            return labels;
        }

        public String GetActionLabel(int index)
        {
            if (index < 0 || index >= Actions.Count)
            {
                MyLog.ERROR.WriteLine("Index ouf of range");
                return "";
            }
            return Actions[index].GetLabel();
        }

        public int GetNoPrimitiveActinos()
        {
            int poc = 0;
            for (int i = 0; i < Actions.Count; i++)
            {
                if (Actions[i].GetLevel() > 0)
                {
                    break;
                }
                poc++;
            }
            return poc;
        }

        public int[] GetPrimitiveActionIndexes()
        {
            List<int> indexes = new List<int>();

            for (int i = 0; i < this.Actions.Count; i++)
            {
                if (Actions[i].GetLevel() == 0)
                {
                    indexes.Add(i);
                }
            }
            return indexes.ToArray();
        }
    }

    public class MyVariable : MyAstractDecisionSpaceUnit
    {
        public static readonly int DEF_VALUE = int.MinValue;
        public List<float> Values{ get; set; }
        public float Current { get; set; }
        private bool m_changed = false;

        // pointer to the StochasticReturnPredictor that controls this variable
        public MyStochasticReturnPredictor MyAction { get; set; }

        public MyVariable(String label, int level)
            : base(label, level)
        {
            this.Current = DEF_VALUE;
            Values = new List<float>();
        }

        public void SetMyAction(MyStochasticReturnPredictor myAction)
        {
            this.MyAction = myAction;
        }

        public void RegisterValue(float val, int myIndex, VariableManager vm)
        {
            // change detected
            if (Current == DEF_VALUE)
            {
                this.m_changed = false;
                this.Current = val;
                Values.Add(val);
            }
            else if (val != Current)
            {
                this.m_changed = true;
                this.Current = val;

                if (!Values.Contains(val))
                {
                    Values.Add(val);
                }
                // if no abstract aciotn is added, rise the request for it
                if (vm.GetVarNo(myIndex).MyAction == null)
                {
                    vm.ShouldBeSubspaced.Add(myIndex);
                }
            }
            else
            {
                this.m_changed = false;
            }
        }

        public bool JustChanged()
        {
            return this.m_changed;
        }
    }

    /// <summary>
    /// Stores all variables that agent have seen so far (list is updated from the
    /// input data).
    /// </summary>
    public class VariableManager
    {
        public IVariableHistory VarHistory{ get; set; }

        // store all values of all variables (those with one value are constants)
        public readonly int MAX_VARIABLES;

        private MyVariable[] m_vars;
        private String[] m_labels;

        // list of variable indexes that should be subspaced
        public List<int> ShouldBeSubspaced{ get; set; } 

        public VariableManager(int maxVariables, MyModuleParams setup)
        {
            if (maxVariables <= 0)
            {
                maxVariables = 1;
            }
            else
            {
                this.MAX_VARIABLES = maxVariables;
            }
            m_labels = new String[this.MAX_VARIABLES];

            for (int i = 0; i < m_labels.Length; i++)
            {
                m_labels[i] = "" + i;
            }
            this.InitVars();
            this.VarHistory = new MyVariableHistory(this, setup);
            this.VarHistory.AddAllPotentialVariables();

            this.ShouldBeSubspaced = new List<int>();
        }

        public VariableManager(String[] labels, MyModuleParams setup)
        {
            this.m_labels = labels;
            m_vars = new MyVariable[labels.Length];

            if (MAX_VARIABLES <= 0)
            {
                MAX_VARIABLES = 1;
            }
            else
            {
                this.MAX_VARIABLES = labels.Length;
            }
            this.InitVars();
            this.VarHistory = new MyVariableHistory(this, setup);
            this.VarHistory.AddAllPotentialVariables();
        }

        private void InitVars()
        {
            this.m_vars = new MyVariable[this.MAX_VARIABLES];

            for (int i = 0; i < m_labels.Length; i++)
            {
                m_vars[i] = new MyVariable(m_labels[i], 0);
            }
        }

        public void MonitorAllInputValues(float[] inputData, ISubspacer subspacer)
        {
            if (inputData.Length > this.MAX_VARIABLES)
            {
                MyLog.ERROR.WriteLine("input data have too many dimensions " + inputData.Length
                    + ", increase maxVariables (current=" + this.MAX_VARIABLES + ")");
                return;
            }
            for (int i = 0; i < inputData.Length; i++)
            {
                m_vars[i].RegisterValue((int)inputData[i], i, this);
            }
            VarHistory.RegisterChanges();
        }

        public MyVariable GetVarNo(int index)
        {
            if (index >= this.MAX_VARIABLES)
            {
                MyLog.ERROR.WriteLine("Index of requested variable out of range!");
                return null;
            }
            return m_vars[index];
        }

        /// <summary>
        /// Decode the current state based on all known variables and their current values.
        /// </summary>
        /// <returns>List of indexes (only states) to be used in the Q(s,a) memory.</returns>
        public int[] GetCurrentState()
        {
            int[] states = new int[MAX_VARIABLES];

            for (int i = 0; i < states.Length; i++)
            {
                states[i] = (int)m_vars[i].Current;
            }
            return states;
        }

        public int GetMaxVariables()
        {
            return this.MAX_VARIABLES;
        }
    }
}
