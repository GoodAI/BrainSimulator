using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;
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

        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            SizeV = new Vector2(20, 20);

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            var avatar = world.GetAvatar(AvatarID);
            PositionCenterV2 = avatar.Position;

            base.Draw(renderer, world);
        }

        #endregion
    }
}
