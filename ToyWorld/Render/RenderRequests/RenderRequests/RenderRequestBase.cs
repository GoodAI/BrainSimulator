using System.Drawing;
using GoodAI.ToyWorld.Control;
using VRageMath;
using RectangleF = VRageMath.RectangleF;

namespace Render.RenderRequests.RenderRequests
{
    public abstract class RenderRequestBase : RenderRequest, IRenderRequest
    {
        #region IRenderRequest overrides

        public System.Drawing.Size Size { get; protected set; }
        public System.Drawing.PointF PositionCenter { get; protected set; }
        public System.Drawing.RectangleF View { get; protected set; }

        // TODO: checks
        public virtual Size Resolution { get; set; }

        #endregion


        protected Vector2I SizeV { get { return (Vector2I)Size; } set { Size = new Size(value.X, value.Y); } }
        protected Vector2 PositionCenterV { get { return (Vector2)PositionCenter; } set { PositionCenter = new PointF(value.X, value.Y); } }
        protected RectangleF ViewV { get { return (RectangleF)View; } set { View = new System.Drawing.RectangleF(value.X, value.Y, value.Width, value.Height); } }
    }
}
