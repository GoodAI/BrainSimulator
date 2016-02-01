using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using GoodAI.BrainSimulator.NodeView;
using GoodAI.Core.Configuration;
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

            DeviceNode.OnKeyDown(e.KeyValue);

            e.Handled = true;
            e.SuppressKeyPress = true;
        }

        public override void OnKeyUp(KeyEventArgs e)
        {
            base.OnKeyUp(e);

            if (e.Handled)
                return;

            DeviceNode.OnKeyUp(e.KeyValue);

            e.Handled = true;
            e.SuppressKeyPress = true;
        }
    }
}
