using System.Drawing;
using GoodAI.ToyWorld.Control;

namespace Render.RenderRequests.AvatarRenderRequests
{
    public abstract class AvatarRRBase : RenderRequest, IAvatarRenderRequest
    {
        protected AvatarRRBase(int avatarID)
        {
            AvatarID = avatarID;
        }


        #region IAvatarRenderRequest overrides

        public float AvatarID { get; protected set; }

        public virtual float Size { get; set; }
        public virtual float Position { get; set; }
        public float RelativePosition { get; set; }
        public virtual Point Resolution { get; set; }

        #endregion
    }
}
