using GoodAI.ToyWorld.Control;
using Render.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class FovAvatarRR : AvatarRRBase, IFovAvatarRR
    {
        #region Genesis

        public FovAvatarRR(int avatarID)
            : base(avatarID)
        { }

        #endregion

        #region IFovAvatarRR overrides

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            SizeV = new Vector2(20, 20);

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            var avatar = world.GetAvatar(AvatarID);
            PositionCenterV = avatar.Position;

            base.Draw(renderer, world);
        }

        #endregion
    }
}
