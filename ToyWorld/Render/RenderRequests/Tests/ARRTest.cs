using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Render.RenderRequests.AvatarRenderRequests;
using Render.Shaders;

namespace Render.RenderRequests.Tests
{
    class ARRTest : AvatarRenderRequestBase, IARRTest
    {
        private GeometryBase m_sq;
        private Shader m_copyShader;


        public ARRTest(int avatarID)
            : base(avatarID)
        { }

        public override void Dispose()
        {
            m_sq.Dispose();
            m_copyShader.Dispose();
        }


        #region IARRTest overrides

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
            GL.Clear(ClearBufferMask.ColorBufferBit);

            m_copyShader.Use(renderer);
            m_sq.Draw();
        }

        #endregion
    }
}
