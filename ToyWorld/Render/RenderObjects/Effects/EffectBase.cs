using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace Render.RenderObjects.Effects
{
    internal class EffectBase : IDisposable
    {
        const string ShaderPathBase = "Render.RenderObjects.Effects.Src.";

        private readonly int m_prog;


        #region Genesis

        // TODO: genericity
        protected EffectBase(string vertPath, string fragPath)
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

        private int LoadShader(string name, ShaderType type)
        {
            var handle = GL.CreateShader(type);

            Stream vertSrc = Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + name);
            Debug.Assert(vertSrc != null);

            StreamReader str = new StreamReader(vertSrc);
            string res = str.ReadToEnd();

            GL.ShaderSource(handle, res);
            GL.CompileShader(handle);

            res = GL.GetShaderInfoLog(handle);

            Debug.Assert(string.IsNullOrEmpty(res), res);

            return handle;
        }

        public virtual void Dispose()
        {
            GL.DeleteProgram(m_prog);
        }

        #endregion


        public void Use()
        {
            GL.UseProgram(m_prog);
        }


        #region Uniforms

        public int GetUniformLocation(string name)
        {
            return GL.GetUniformLocation(m_prog, name);
        }


        public void SetUniform1(int pos, int val)
        {
            GL.Uniform1(pos, val);
        }

        public void SetUniform3(int pos, Vector3I val)
        {
            GL.Uniform3(pos, val.X, val.Y, val.Z);
        }

        public void SetUniform4(int pos, Vector4I val)
        {
            GL.Uniform4(pos, val.X, val.Y, val.Z, val.W);
        }

        // TODO: Jde matrix prevest na array bez kopirovani a unsafe kodu?
        /// <summary>
        ///Passed matrices are applied from left to right (as in vert*(a*b*c) -- a will be first).
        /// </summary>
        public void SetUniformMatrix4(int pos, Matrix val)
        {
            unsafe
            {
                GL.UniformMatrix4(pos, 1, false, val.M.data);
            }
        }

        #endregion
    }
}
