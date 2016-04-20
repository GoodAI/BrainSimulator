using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    public abstract class AvatarRRBase : RenderRequest, IAvatarRenderRequest
    {
        protected Vector2 RelativePositionV { get; set; }


        protected AvatarRRBase(int avatarID)
        {
            AvatarID = avatarID;
        }


        #region IAvatarRenderRequest overrides

        public int AvatarID { get; protected set; }

        public uint[] Image { get; private set; }

        public System.Drawing.PointF RelativePosition
        {
            get { return new System.Drawing.PointF(RelativePositionV.X, RelativePositionV.Y); }
            set { RelativePositionV = (Vector2)value; }
        }

        #endregion

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            Image = new uint[Resolution.Width * Resolution.Height];

            base.Init(renderer, world);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            PositionCenterV += RelativePositionV;

            base.Draw(renderer, world);

            // Gather data to host mem
            GL.ReadPixels(0, 0, Resolution.Width, Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, Image);
        }
    }
}
