using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderRequests.RenderRequests;
using Render.Tests.Effects;
using Render.Tests.Geometries;

namespace Render.Tests.RRs
{
    public interface IBasicTexRR : IRenderRequest
    { }

    class BasicTexRR : RenderRequestBase, IBasicTexRR
    {
        private bool odd;


        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            GL.ClearColor(Color.Black);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            Matrix4 m;

            if (odd = !odd)
                m = Matrix4.CreateScale(0.5f);
            else
                m = Matrix4.CreateScale(0.1f);


            renderer.EffectManager.Use<NoEffect>();
            renderer.GeometryManager.Draw<FancyFullscreenQuad>();
        }

        #endregion
    }
}
