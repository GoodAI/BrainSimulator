using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using GoodAI.BrainSimulator.NodeView;
using GoodAI.Core.Configuration;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using Graph;

namespace GoodAI.BrainSimulator.Nodes
{
    internal class DeviceInputView : MyNodeView
    {
        public DeviceInputView(MyNodeConfig nodeConfig, GraphControl owner) : base(nodeConfig, owner)
        {
        }

        private DeviceInput DeviceNode
        {
            get { return Node as DeviceInput; }
        }

        public override void OnKeyDown(KeyEventArgs e)
        {
            base.OnKeyDown(e);

            if (e.Handled)
                return;

            if (Node.Owner.SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED)
                return;

            DeviceNode.SetKeyDown(e.KeyValue);

            e.Handled = true;
            e.SuppressKeyPress = true;
        }

        public override void OnKeyUp(KeyEventArgs e)
        {
            base.OnKeyUp(e);

            if (e.Handled)
                return;

            if (Node.Owner.SimulationHandler.State == MySimulationHandler.SimulationState.STOPPED)
                return;

            DeviceNode.SetKeyUp(e.KeyValue);

            e.Handled = true;
            e.SuppressKeyPress = true;
        }
    }
}
