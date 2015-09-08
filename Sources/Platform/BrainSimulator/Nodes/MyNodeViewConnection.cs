using GoodAI.BrainSimulator.NodeView;
using GoodAI.Core;
using Graph;

namespace GoodAI.BrainSimulator.Nodes
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
