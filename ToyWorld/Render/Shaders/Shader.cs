using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;

namespace Render.Shaders
{
    internal class Shader : IDisposable
    {
        const string ShaderPathBase = "Render.Shaders.Src.";

        private readonly int m_prog;


        // TODO: genericity
        public Shader(string vertPath, string fragPath)
        {
            int vert = LoadShader(vertPath, ShaderType.VertexShader);
            int frag = LoadShader(fragPath, ShaderType.FragmentShader);

            m_prog = GL.CreateProgram();
            GL.AttachShader(m_prog, vert);
            GL.AttachShader(m_prog, frag);
            GL.LinkProgram(m_prog);

            GL.DeleteShader(vert);
            GL.DeleteShader(frag);

            var res = GL.GetProgramInfoLog(m_prog);

            Debug.Assert(string.IsNullOrEmpty(res), res);
        }

        int LoadShader(string name, ShaderType type)
        {
            var handle = GL.CreateShader(type);
            var vertSrc = Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + name);

            Debug.Assert(vertSrc != null);

            var str = new StreamReader(vertSrc);
            string res = str.ReadToEnd();

            GL.ShaderSource(handle, res);
            GL.CompileShader(handle);

            res = GL.GetShaderInfoLog(handle);

            Debug.Assert(string.IsNullOrEmpty(res), res);

            return handle;
        }

        public void Dispose()
        {
            GL.DeleteProgram(m_prog);
        }


        public void Use(RendererBase renderer)
        {
            if (renderer.CurrentShader == this)
                return;

            GL.UseProgram(m_prog);
            renderer.CurrentShader = this;
        }
    }
}
