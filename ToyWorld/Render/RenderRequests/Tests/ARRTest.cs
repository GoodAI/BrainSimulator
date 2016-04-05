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

namespace Render.RenderRequests.Tests
{
    class ARRTest : AvatarRenderRequestBase, IARRTest
    {
        private readonly Square m_sq = new Square();
        private int m_prog;


        public ARRTest(int avatarID)
            : base(avatarID)
        { }


        #region IARRTest overrides

        public float MemAddress { get; set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(IRenderer renderer)
        {
            GL.ClearColor(Color.Black);

            m_sq.Init();

            renderer.Window.Visible = true;

            // Init shaders
            int vert = LoadShader("Render.Shaders.Basic.vert", ShaderType.VertexShader);
            int frag = LoadShader("Render.Shaders.Basic.frag", ShaderType.FragmentShader);

            m_prog = GL.CreateProgram();
            GL.AttachShader(m_prog, vert);
            GL.AttachShader(m_prog, frag);
            GL.LinkProgram(m_prog);

            var res = GL.GetProgramInfoLog(m_prog);

            Debug.Assert(string.IsNullOrEmpty(res), res);

            GL.UseProgram(m_prog);
        }

        int LoadShader(string name, ShaderType type)
        {
            var handle = GL.CreateShader(type);
            var vertSrc = Assembly.GetExecutingAssembly().GetManifestResourceStream(name);

            Debug.Assert(vertSrc != null);

            var str = new StreamReader(vertSrc);
            string res = str.ReadToEnd();

            GL.ShaderSource(handle, res);
            GL.CompileShader(handle);

            res = GL.GetShaderInfoLog(handle);

            Debug.Assert(string.IsNullOrEmpty(res), res);

            return handle;
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);


            renderer.Context.SwapBuffers();
        }

        #endregion
    }
}
