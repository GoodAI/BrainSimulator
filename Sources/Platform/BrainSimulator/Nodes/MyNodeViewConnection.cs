using GoodAI.BrainSimulator.NodeView;
using GoodAI.Core;
using GoodAI.Core.Nodes;
using Graph;

namespace GoodAI.BrainSimulator.Nodes
{
    public class MyNodeViewConnection : NodeConnection
    {
        public override string Name
        {
            get
            {
                if (From != null && (From.Node is MyNodeView) && (Tag is MyConnection))
                {
                    MyNodeView fromNodeView = From.Node as MyNodeView;
                    
                    var memBlock = fromNodeView.Node.GetAbstractOutput((Tag as MyConnection).FromIndex);
                    if (memBlock != null) 
                    {
                        if (!Hidden)
                        {
                            return memBlock.Dims.Print();
                        }
                        else
                        {
                            if (fromNodeView.Node is MyParentInput)
                            {
                                return memBlock.Dims.Print() + ", " + (memBlock.Owner is MyWorld ? "World" : "Group") + ": " + memBlock.Name;
                            }
                            else
                            {
                                return memBlock.Dims.Print() + ", " + fromNodeView.Node.Name + ": " + memBlock.Name;
                            }
                        }
                    }                                      
                }

                return "0";
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

        public bool Hidden
        {
            get { return (state & RenderState.Hidden) != 0; }
            set
            {
                if (value)
                {
                    state |= RenderState.Hidden;
                }
                else
                {
                    state &= ~RenderState.Hidden;
                }
            }
        }
    }
}
