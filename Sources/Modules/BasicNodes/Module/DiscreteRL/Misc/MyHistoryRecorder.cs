using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;

namespace GoodAI.Modules.Harm
{
    public interface IHistoryRecorder
    {
        void storeActionVariables(int executedActionIndex);
    }

    public interface IVariableHistory
    {
        /// <summary>
        /// Add a variable to the history
        /// </summary>
        /// <param name="index">index of the variable in the RootDecisionSpace</param>
        void AddVariable(int index);

        void RemoveVariable(int index);

        /// <summary>
        /// Remembers values of all added variables in time (or all possible variables discovered in the future)
        /// </summary>
        void AddAllPotentialVariables();

        void RegisterChanges();

        int GetVariableChangedBefore(int steps);
        List<int> GetVariablesChangedBefore(int steps);
    }

    public interface IActionHistory
    {
        void AddAction(int index);

        void RemoveAction(int index);

        void RegisterExecutedAction(int index);

        void AddAllCurrentActions();

        int GetActionExecutedBefore(int steps);
    }

    public class MyActionHistory : IActionHistory
    {
        private MyModuleParams setup;
        private List<int> changed;
        protected Dictionary<int, bool> isMonitored;
        private ActionManager am;

        public MyActionHistory(ActionManager am, MyModuleParams setup)
        {
            this.setup = setup;
            this.am = am;
            this.changed = new List<int>();
            this.isMonitored = new Dictionary<int, bool>();
        }

        public int GetActionExecutedBefore(int steps)
        {
            if (steps < 0 || steps >= changed.Count)
            {
                return -1;
            }
            return changed[steps];
        }

        public void AddAction(int index)
        {
            if (index >= am.GetNoActions() || index < 0)
            {
                MyLog.ERROR.WriteLine("index out of range");
                return;
            }
            if (!isMonitored.ContainsKey(index))
            {
                isMonitored.Add(index, true);
            }
        }

        public void RemoveAction(int index)
        {
            if (index >= am.GetNoActions() || index < 0)
            {
                MyLog.ERROR.WriteLine("index out of range");
                return;
            }
            if (!isMonitored.ContainsKey(index))
            {
                MyLog.ERROR.WriteLine("trying to remove action that is not monitored");
                return;
            }
            isMonitored.Remove(index);
        }

        public void AddAllCurrentActions()
        {
            isMonitored.Clear();
            for (int i = 0; i < am.GetNoActions(); i++)
            {
                isMonitored.Add(i, true);
            }
        }

        public void RegisterExecutedAction(int index)
        {
            if (!this.isMonitored[index])
            {
                MyLog.DEBUG.WriteLine("Warning: you want to register action that is not monitored");
                return;
            }
            this.PushNewChange(index);
        }

        private void PushNewChange(int action)
        {
            while (setup.ActionLength > 0 && changed.Count >= setup.ActionLength)
            {
                changed.RemoveAt(changed.Count - 1);
            }
            changed.Insert(0, action);
        }
    }

    /// <summary>
    /// The same as a VariableHistory, but each SRP has its own.
    /// </summary>
    public class MyLocalVariableHistory
    {
        int INIT_VAL = 1;
        float addVal = 1f;

        Dictionary<int, float> changes;
        MyModuleParams m_setup;
        MyRootDecisionSpace m_rds;

        public MyLocalVariableHistory(MyRootDecisionSpace rds,
            MyModuleParams setup, IDecisionSpace ds)
        {
            m_rds = rds;
            m_setup = setup;
            changes = new Dictionary<int, float>();

            // all potential variables added by default
            for (int i = 0; i < rds.VarManager.GetMaxVariables(); i++)
            {
                changes.Add(i, INIT_VAL);
            }
        }

        /// <summary>
        /// This updates the variable weights in the DS, if the weight is under the threshold, 
        /// the corresponding variable should be removed from the DS.
        /// </summary>
        /// <param name="parent"></param>
        public void monitorVariableChanges(MyStochasticReturnPredictor parent)
        {
            // update info about changed variables
            List<int> changed = m_rds.VarManager.VarHistory.GetVariablesChangedBefore(1);
            if (changed != null)
            {
                for (int i = 0; i < changed.Count; i++)
                {
                    changes[changed[i]] += addVal;
                }
            }

            // decay all that are included in this DS
            for (int i = 0; i < m_rds.VarManager.GetMaxVariables(); i++)
            {
                if (parent.Ds.IsVariableIncluded(i)
                    && m_rds.VarManager.GetVarNo(i).Values.Count > 1
                    && i != parent.GetPromotedVariableIndex())
                {
                    changes[i] -= m_setup.OnlineHistoryForgettingRate;
                }
            }
        }

        /// <summary>
        /// This should be called only after receiving the reward. 
        /// Frequency of variable changes relative to rewards received.
        /// </summary>
        /// <param name="parent"></param>
        public void performOnlineVariableRemoving(MyStochasticReturnPredictor parent)
        {
            bool removed = false;
            // decay all that are included in this DS
            for (int i = 0; i < m_rds.VarManager.GetMaxVariables(); i++)
            {
                if (parent.Ds.IsVariableIncluded(i)
                    && m_rds.VarManager.GetVarNo(i).Values.Count > 1
                    && i != parent.GetPromotedVariableIndex())
                {
                    if (changes[i] < m_setup.OnlineVariableRemovingThreshold)
                    {
                        removed = true; parent.Ds.RemoveVariable(i);
                    }
                }
            }
            if (removed)
            {
                String output =
                    "SRP: " + parent.GetLabel() + ": Variables removed, current ones: ";
                
                for (int i = 0; i < m_rds.VarManager.GetMaxVariables(); i++)
                {
                    if (parent.Ds.IsVariableIncluded(i))
                    {
                        output += i + ", ";
                    }
                }
                MyLog.DEBUG.WriteLine(output);
            }
        }
    }

    public class MyVariableHistory : IVariableHistory
    {
        private List<List<int>> changed;

        protected bool[] isMonitored;
        private VariableManager vm;
        protected MyModuleParams setup;

        public MyVariableHistory(VariableManager vm, MyModuleParams setup)
        {
            this.vm = vm;
            this.setup = setup;

            this.changed = new List<List<int>>();

            isMonitored = new bool[vm.GetMaxVariables()];
        }

        /// <summary>
        /// Looks n steps into the history, checks for THE FIRST monitored variable that has changed.
        /// </summary>
        /// <param name="steps">number steps in the past</param>
        /// <returns>first identified variable that changed its value</returns>
        public int GetVariableChangedBefore(int steps)
        {
            if (steps < 0 || steps >= changed.Count)
            {
                return -1;
            }
            if (changed[steps].Count == 0)
            {
                return -1;
            }

            for (int i = 0; i < changed[steps].Count; i++)
            {
                if (this.isMonitored[changed[steps][i]])
                {
                    return changed[steps][i];
                }
            }
            return -1;
        }

        /// <summary>
        /// Return list of monitored variables that changed given no. of steps in the hsitory
        /// </summary>
        /// <param name="steps">number steps in the past (0 excluded)</param>
        /// <returns>list of all variable indexes that changed that time step</returns>
        public List<int> GetVariablesChangedBefore(int steps)
        {
            if (steps < 0 || steps >= changed.Count)
            {
                return null;
            }
            if (changed[steps].Count == 0)
            {
                return null;
            }

            List<int> ch = new List<int>();

            for (int i = 0; i < changed[steps].Count; i++)
            {
                if (this.isMonitored[changed[steps][i]])
                {
                    ch.Add(changed[steps][i]);
                }
            }
            return ch;
        }

        public void AddAllPotentialVariables()
        {
            for (int i = 0; i < vm.GetMaxVariables(); i++)
            {
                isMonitored[i] = true;
            }
        }

        public void AddVariable(int index)
        {
            if (index >= this.isMonitored.Length || index < 0)
            {
                MyLog.ERROR.WriteLine("index out of range");
            }
            this.isMonitored[index] = true;
        }

        public void RemoveVariable(int index)
        {
            if (index >= this.isMonitored.Length || index < 0)
            {
                MyLog.ERROR.WriteLine("index out of range");
            }
            this.isMonitored[index] = false;
        }

        public void RegisterChanges()
        {
            List<int> changedNow = new List<int>();
            for (int i = 0; i < this.isMonitored.Length; i++)
            {
                if (this.isMonitored[i])
                {
                    if (vm.GetVarNo(i).JustChanged())
                    {
                        changedNow.Add(i);
                    }
                }
            }
            this.PushNew(changedNow);
        }

        private void PushNew(List<int> changedNow)
        {
            while (setup.VarLength > 0 && changed.Count >= setup.VarLength)
            {
                changed.RemoveAt(changed.Count - 1);
            }
            changed.Insert(0, changedNow);
        }
    }
}
