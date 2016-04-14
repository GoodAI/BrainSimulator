using GoodAI.ToyWorld.Control;
using VRageMath;
using RectangleF = VRageMath.RectangleF;

namespace Render.RenderRequests
{
    public abstract class AvatarRRBase : RenderRequest, IAvatarRenderRequest
    {
        protected AvatarRRBase(int avatarID)
        {
            AvatarID = avatarID;
        }


        #region IAvatarRenderRequest overrides

        public float AvatarID { get; protected set; }

        public virtual System.Drawing.Size Size { get; set; }
        public System.Drawing.PointF PositionCenter { get; private set; }
        public System.Drawing.PointF RelativePosition { get; set; }
        public System.Drawing.RectangleF View { get; private set; }

        public virtual System.Drawing.Size Resolution { get; set; }

        #endregion
 

        protected Vector2I SizeV { get { return (Vector2I)Size; } set { Size = new System.Drawing.Size(value.X, value.Y); } }
        protected Vector2 PositionCenterV { get { return (Vector2)PositionCenter; } set { PositionCenter = new System.Drawing.PointF(value.X, value.Y); } }
        protected RectangleF ViewV { get { return (RectangleF)View; } set { View = new System.Drawing.RectangleF(value.X, value.Y, value.Width, value.Height); } }
    }
}
