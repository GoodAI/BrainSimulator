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
                string name = "0";

                if (From != null && (From.Node is MyNodeView) && (Tag is MyConnection))
                {
                    MyNodeView fromNodeView = From.Node as MyNodeView;

                    var memBlock = fromNodeView.Node.GetAbstractOutput((Tag as MyConnection).FromIndex);
                    if (memBlock != null)
                        name = memBlock.Dims.Print(hideTrailingOnes: true);
                }

                var connection = Tag as MyConnection;
                if (connection != null && connection.IsLowPriority)
                    name = name + " (low priority)";

                return name;
            }
            set
            {

            }
        }

        public bool Backward
        {
            get { return (state & RenderState.Backward) != 0; }
            set
            {
                if (value)
                {
                    state |= RenderState.Backward;
                }
                else
                {
                    state &= ~RenderState.Backward;
                }
            }
        }

        public bool Dynamic
        {
            get { return (state & RenderState.Marked) != 0; }
            set
            {
                if (value)
                {
                    state |= RenderState.Marked;
                }
                else
                {
                    state &= ~RenderState.Marked;
                }
            }
        }
    }
}
