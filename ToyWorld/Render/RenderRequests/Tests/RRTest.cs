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

namespace Render.RenderRequests.Tests
{
    class RRTest : RenderRequestBase, IRRTest
    {
        private readonly Square m_sq = new Square();
        private int m_prog;

        private bool odd;


        public RRTest()
        {
            WindowKeypressResult = default(Key);
        }

        public override void Dispose()
        {
            GL.UseProgram(0);
            GL.DeleteProgram(m_prog);

            m_sq.Dispose();
        }

        #region IRRTest overrides

        public Key WindowKeypressResult { get; private set; }

        public float MemAddress { get; set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(IRenderer renderer)
        {
            GL.ClearColor(Color.Black);

            m_sq.Init();

            renderer.Window.KeyDown += (sender, args) => WindowKeypressResult = args.Key;
            renderer.Window.Visible = true;


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
            DrawInternal(renderer);
            HandleWindow(renderer);
        }

        void DrawInternal(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            Matrix4 m;

            if (odd = !odd)
                m = Matrix4.CreateScale(0.5f);
            else
                m = Matrix4.CreateScale(0.1f);


            m_sq.Draw();

            renderer.Context.SwapBuffers();
        }

        void HandleWindow(RendererBase renderer)
        {
            renderer.Window.ProcessEvents();
        }

        #endregion
    }
}
