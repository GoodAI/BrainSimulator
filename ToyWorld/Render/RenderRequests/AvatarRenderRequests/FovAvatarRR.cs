using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
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

        public uint[] Image { get; private set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            Image = new uint[Resolution.Width * Resolution.Height];

            SizeV = new Vector2(20, 20);

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            // Setup params
            var avatar = world.GetAvatar(AvatarID);
            PositionCenterV = new Vector3(avatar.Position, 0);

            base.Draw(renderer, world);

            // Gather data to host mem
            GL.ReadPixels(0, 0, Resolution.Width, Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, Image);
        }

        #endregion
    }
}
