using GoodAI.ToyWorld.Control;

namespace Render.RenderRequests.AvatarRenderRequests
{
    public abstract class AvatarRenderRequestBase : RenderRequest, IAvatarRenderRequest
    {
        protected AvatarRenderRequestBase(int avatarID)
        {
            AvatarID = avatarID;
        }


        #region IAvatarRenderRequest overrides

        public float AvatarID { get; protected set; }

        public virtual float Size { get; protected set; }
        public virtual float Position { get; set; }
        public virtual float Resolution { get; set; }

        #endregion
    }
}
