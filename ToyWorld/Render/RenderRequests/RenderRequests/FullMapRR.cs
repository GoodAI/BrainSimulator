using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderRequests.AvatarRenderRequests;
using Render.Tests.Effects;
using Render.Tests.Geometries;

namespace Render.RenderRequests.RenderRequests
{
    internal class FullMapRR : RenderRequestBase, IFullMapRenderRequest
    {

        #region IAvatarRenderRequestFoV overrides

        public uint[] Image { get; protected set; }

        #endregion

        #region AvatarRenderRequestBase overrides

        public override float Size { get { return Image.Length; } }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            //m_pbo = new VBO(renderer.Window.Width * renderer.Window.Height, target: BufferTarget.PixelPackBuffer, hint: BufferUsageHint.StreamRead);

            // TODO: mel by mit vlastni rendertarget s custom dims, spravovanej nejakym managerem
            Image = new uint[renderer.Width * renderer.Height];

            GL.ClearColor(Color.Black);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            var effect = renderer.EffectManager.Get<NoEffect>();
            renderer.EffectManager.Use(effect);
            renderer.GeometryManager.Get<FancyFullscreenQuad>().Draw();

            GL.ReadPixels(0, 0, renderer.Width, renderer.Height, PixelFormat.Rgba, PixelType.UnsignedByte, Image);
        }

        #endregion
    }
}
