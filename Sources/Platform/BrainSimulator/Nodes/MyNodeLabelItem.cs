using GoodAI.Core.Nodes;
using Graph.Items;
using System.Drawing;

namespace GoodAI.BrainSimulator.Nodes
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
