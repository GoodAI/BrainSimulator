using BrainSimulator.Configuration;
using BrainSimulator.Nodes;
using BrainSimulator.Utils;
using Graph;
using Graph.Items;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulatorGUI.NodeView
{
    internal class MyGateInputView : MyNodeView
    {
        private NodeSliderItem slider;

        public MyGateInputView(MyNodeConfig nodeInfo, GraphControl owner) : base(nodeInfo, owner) { }

        public override void UpdateView()
        {
            base.UpdateView();       

            if (slider == null && Node != null)
            {
                slider = new NodeSliderItem(null, 0, 0, 0, 1, (Node as MyGateInput).GetWeight(), false, false);
                slider.ValueChanged += slider_ValueChanged;
                AddItem(slider);
            }            
        }

        void slider_ValueChanged(object sender, NodeItemEventArgs e)
        {
            NodeSliderItem slider = (NodeSliderItem)sender;            

            if (Node is MyGateInput)
            {
                (Node as MyGateInput).SetWeight((e.Item as NodeSliderItem).Value);                
            }
        }
    }
}
