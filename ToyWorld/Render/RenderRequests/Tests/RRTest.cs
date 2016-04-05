using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Render.RenderRequests.RenderRequests;
using Render.Shaders;

namespace Render.RenderRequests.Tests
{
    class RRTest : RenderRequestBase, IRRTest
    {
        private GeometryBase m_sq;
        private Shader m_copyShader;

        private bool odd;


        public override void Dispose()
        {
            m_sq.Dispose();
            m_copyShader.Dispose();
        }


        #region IRRTest overrides

        public float MemAddress { get; set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(IRenderer renderer)
        {
            GL.ClearColor(Color.Black);
            renderer.Window.Visible = true;

            m_sq = new FancyFullscreenQuad();
            m_copyShader = new Shader("Basic.vert", "Basic.frag");
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            Matrix4 m;

            if (odd = !odd)
                m = Matrix4.CreateScale(0.5f);
            else
                m = Matrix4.CreateScale(0.1f);


            m_copyShader.Use(renderer);
            m_sq.Draw();

            renderer.Context.SwapBuffers();
        }

        #endregion
    }
}
