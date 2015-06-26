using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulatorGUI.NodeView;
using Graph;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulatorGUI.Nodes
{
    public class MyNodeViewConnection : NodeConnection
    {
        public override string Name
        {
            get
            {
                if (From != null && From.Node is MyNodeView)
                {
                    MyNodeView fromNodeView = From.Node as MyNodeView;
                    return fromNodeView.Node.GetOutputSize((Tag as MyConnection).FromIndex) + "";
                }
                else return "0";
            }
            set
            {
                
            }
        }
    }
}
