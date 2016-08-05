using GoodAI.BrainSimulator.Nodes;
using GoodAI.Core.Configuration;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using Graph;
using Graph.Items;
using System;
using System.Linq;
using System.Reflection;

namespace GoodAI.BrainSimulator.NodeView
{
    internal class MyVariableBranchView : MyNodeView
    {
        private bool m_isUpdatingBranches = false;

        public MyVariableBranchView(MyNodeConfig nodeInfo, GraphControl owner) : base(nodeInfo, owner) { }

        protected virtual void AddInputBranch()
        {
            string name = null;

            if (Node is IMyVariableBranchViewWithNamesNode)
            {
                name = ((IMyVariableBranchViewWithNamesNode)Node).GetInputBranchName(m_inputBranches.Count);
            }
            if (name == null)
            {
                name = "Input " + (m_inputBranches.Count + 1);
            }

            NodeLabelItem branch = new NodeLabelItem(name, true, false)
            {
                Tag = m_inputBranches.Count,
                IsPassive = true
            };

            m_inputBranches.Add(branch);
            AddItem(branch);
        }

        protected void RemoveInputBranch()
        {
            NodeItem branch = m_inputBranches.Last();

            if (branch.Input.HasConnection)
            {
                foreach (NodeConnection nc in branch.Input.Connectors.ToList())
                {
                    Owner.Disconnect(nc);
                }
            }

            m_inputBranches.Remove(branch);
            RemoveItem(branch);
        }

        protected virtual void AddOutputBranch()
        {
            string name = null;

            if (Node is IMyVariableBranchViewWithNamesNode)
            {
                name = ((IMyVariableBranchViewWithNamesNode)Node).GetOutputBranchName(m_outputBranches.Count);
            }
            if (name == null)
            {
                name = "Output " + (m_outputBranches.Count + 1);
            }

            NodeLabelItem branch = new NodeLabelItem(name, false, true)
            {
                Tag = m_outputBranches.Count,
                IsPassive = true
            };
            m_outputBranches.Add(branch);
            AddItem(branch);
        }

        private string GetOutputBranchName(int index)
        {
            string name = null;

            if (Node is IMyVariableBranchViewWithNamesNode)
            {
                name = ((IMyVariableBranchViewWithNamesNode)Node).GetOutputBranchName(index);
            }
            if (name == null)
            {
                name = "Output " + (index);
            }
            return name;
        }

        private string GetInputBranchName(int index)
        {
            string name = null;

            if (Node is IMyVariableBranchViewWithNamesNode)
            {
                name = ((IMyVariableBranchViewWithNamesNode)Node).GetInputBranchName(index);
            }
            if (name == null)
            {
                name = "Input " + index;
            }
            return name;
        }

        protected void RemoveOutputBranch()
        {
            NodeItem branch = m_outputBranches.Last();

            if (branch.Output.HasConnection)
            {
                foreach (NodeConnection nc in branch.Output.Connectors.ToList())
                {
                    Owner.Disconnect(nc);
                }
            }

            m_outputBranches.Remove(branch);
            RemoveItem(branch);
        }

        protected override void InitBranches()
        {
            MyNodeInfo nodeInfo = Node.GetInfo();

            foreach (PropertyInfo signalInfo in nodeInfo.RegisteredSignals)
            {
                MySignal signal = (MySignal)signalInfo.GetValue(Node);

                MySignalItem signalItem = new MySignalItem(signal);

                m_signals.Add(signalItem);
                AddItem(signalItem);
            }

            for (int i = 0; i < Node.InputBranches; i++)
            {
                AddInputBranch();
            }

            for (int i = 0; i < Node.OutputBranches; i++)
            {
                AddOutputBranch();
            }
        }

        public void UpdateBranches()
        {
            if (m_isUpdatingBranches) // prevent nested calls when disconnecting inputs
            {
                return;
            }
            m_isUpdatingBranches = true;
            int inputChange = m_inputBranches.Count - Node.InputBranches;

            for (int i = 0; i < inputChange; i++)
            {
                RemoveInputBranch();
            }

            for (int i = 0; i < -inputChange; i++)
            {
                AddInputBranch();
            }

            int outputChange = m_outputBranches.Count - Node.OutputBranches;

            for (int i = 0; i < outputChange; i++)
            {
                RemoveOutputBranch();
            }

            for (int i = 0; i < -outputChange; i++)
            {
                AddOutputBranch();
            }

            // potentially all branch labels could change
            if ((inputChange != 0 || outputChange != 0) && Node is IMyVariableBranchViewWithNamesNode)
            {
                IMyVariableBranchViewWithNamesNode node = (IMyVariableBranchViewWithNamesNode)Node;

                for (int i = 0; i < m_inputBranches.Count; i++)
                {
                    ((NodeLabelItem)m_inputBranches[i]).Text = node.GetInputBranchName(i);
                }
                for (int i = 0; i < m_outputBranches.Count; i++)
                {
                    ((NodeLabelItem)m_outputBranches[i]).Text = node.GetOutputBranchName(i);
                }
            }
            m_isUpdatingBranches = false;
        }

        public override void UpdateView()
        {
            base.UpdateView();

            if (BranchChangeNeeded)
                UpdateBranches();
        }
    }

    internal class MyNodeGroupView : MyVariableBranchView
    {
        public MyNodeGroupView(MyNodeConfig nodeInfo, GraphControl owner) : base(nodeInfo, owner) { }

        protected override void AddInputBranch()
        {
            NodeLabelItem branch = new MyNodeLabelItem((Node as MyNodeGroup).GroupInputNodes[m_inputBranches.Count], true, false);
            branch.Tag = m_inputBranches.Count;
            branch.IsPassive = true;

            m_inputBranches.Add(branch);
            AddItem(branch);
        }

        protected override void AddOutputBranch()
        {
            NodeLabelItem branch = new MyNodeLabelItem((Node as MyNodeGroup).GroupOutputNodes[m_outputBranches.Count], false, true);
            branch.Tag = m_outputBranches.Count;
            branch.IsPassive = true;

            m_outputBranches.Add(branch);
            AddItem(branch);
        }
    }
}
