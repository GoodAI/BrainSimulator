using BrainSimulator.Nodes;
using Graph.Items;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulatorGUI.Nodes
{
    internal class MyNodeLabelItem : NodeLabelItem
    {
        protected MyNode m_target;

        public MyNodeLabelItem(MyNode target, bool inputEnabled, bool outputEnabled) :
			base("", inputEnabled, outputEnabled)
		{
            m_target = target;
		}

        public override string Text
        {
            get
            {
                if (internalText != m_target.Name)
                {                    
                    internalText = m_target.Name;
                    TextSize = Size.Empty;
                }
                return internalText;
            }
            set
            {
                
            }
        }
    }
}
