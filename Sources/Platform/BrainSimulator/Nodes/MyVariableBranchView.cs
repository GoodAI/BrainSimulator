using GoodAI.BrainSimulator.Nodes;
using GoodAI.Core.Configuration;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using Graph;
using Graph.Items;
using System.Linq;
using System.Reflection;

namespace GoodAI.BrainSimulator.NodeView
{
    internal class MyVariableBranchView : MyNodeView
    {
        public MyVariableBranchView(MyNodeConfig nodeInfo, GraphControl owner) : base(nodeInfo, owner) { }

        protected virtual void AddInputBranch()
        {
            NodeLabelItem branch = new NodeLabelItem("Input " + (m_inputBranches.Count + 1), true, false)
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
            NodeLabelItem branch = new NodeLabelItem("Output " + (m_outputBranches.Count + 1), false, true)
            {
                Tag = m_outputBranches.Count,
                IsPassive = true
            };

            m_outputBranches.Add(branch);
            AddItem(branch);
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
            NodeLabelItem branch = new MyNodeLabelItem((Node as MyNodeGroup).GroupInputNodes[m_inputBranches.Count] , true, false);
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
