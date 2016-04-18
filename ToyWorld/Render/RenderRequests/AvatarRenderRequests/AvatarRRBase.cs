using System.Drawing;
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

        public int AvatarID { get; protected set; }
        public System.Drawing.PointF RelativePosition { get; set; }

        #endregion
    }
}
