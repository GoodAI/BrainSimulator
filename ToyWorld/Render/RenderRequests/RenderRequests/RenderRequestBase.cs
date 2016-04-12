using System.Drawing;
using GoodAI.ToyWorld.Control;

namespace Render.RenderRequests.RenderRequests
{
    public abstract class RenderRequestBase : RenderRequest, IRenderRequest
    {
        #region IRenderRequest overrides

        public virtual Size Size { get; protected set; }
        public virtual PointF PositionCenter { get; protected set; }
        public virtual Size Resolution { get; set; }

        #endregion
    }
}
