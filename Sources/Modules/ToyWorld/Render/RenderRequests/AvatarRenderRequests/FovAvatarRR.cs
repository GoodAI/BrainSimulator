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
        {
            SizeV = new Vector2(20, 20);
        }

        #endregion

        #region IFovAvatarRR overrides
        #endregion

        #region RenderRequestBase overrides

        public override void Update()
        {
            var avatar = World.GetAvatar(AvatarID);
            PositionCenterV2 = avatar.Position;

            base.Update();
        }

        #endregion
    }
}
