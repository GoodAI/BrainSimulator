using GoodAI.ToyWorld.Control;
using Render.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class FofAvatarRR : AvatarRRBase, IFofAvatarRR
    {
        public Vector2 RelativePositionV { get; set; }


        #region Genesis

        public FofAvatarRR(int avatarID)
            : base(avatarID)
        { }


        #endregion

        #region IFofAvatarRR overrides

        public System.Drawing.PointF RelativePosition
        {
            get { return new System.Drawing.PointF(RelativePositionV.X, RelativePositionV.Y); }
            set { RelativePositionV = (Vector2)value; }
        }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            SizeV = new Vector2(3, 3);

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            // Setup params
            var avatar = world.GetAvatar(AvatarID);
            PositionCenterV = avatar.Position + RelativePositionV;

            base.Draw(renderer, world);

        }

        #endregion
    }
}
