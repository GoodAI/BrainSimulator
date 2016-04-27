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
        protected const string ShaderPathBase = "Render.RenderObjects.Effects.Src.";

        private readonly int m_prog;


        #region Genesis

        // TODO: genericity
        // Addenda serve as switchable extensions to base shaders -- can be used as a different implementation of functions defined in base shaders.
        protected EffectBase(string vertPath, string fragPath, Stream vertAddendum = null, Stream fragAddendum = null)
        {
            m_prog = GL.CreateProgram();

            // Vertex shader
            LoadAndAttachShader(ShaderType.VertexShader, GetShaderSource(vertPath));
            if (vertAddendum != null)
                LoadAndAttachShader(ShaderType.VertexShader, GetShaderSource(vertAddendum));

            // Fragment shader
            LoadAndAttachShader(ShaderType.FragmentShader, GetShaderSource(fragPath));
            if (fragAddendum != null)
                LoadAndAttachShader(ShaderType.FragmentShader, GetShaderSource(fragAddendum));

            GL.LinkProgram(m_prog);

            var res = GL.GetProgramInfoLog(m_prog);
            Debug.Assert(string.IsNullOrEmpty(res), res);
        }

        private string GetShaderSource(string name)
        {
            Stream sourceStream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ShaderPathBase + name);
            Debug.Assert(sourceStream != null);
            return GetShaderSource(sourceStream);
        }

        private string GetShaderSource(Stream sourceStream)
        {
            using (StreamReader reader = new StreamReader(sourceStream))
                return reader.ReadToEnd();
        }

        private void LoadAndAttachShader(ShaderType type, string source)
        {
            var handle = GL.CreateShader(type);

            GL.ShaderSource(handle, source);
            GL.CompileShader(handle);

            string err = GL.GetShaderInfoLog(handle);
            Debug.Assert(string.IsNullOrEmpty(err), err);

            GL.AttachShader(m_prog, handle);
            GL.DeleteShader(handle);
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

        public void SetUniform4(int pos, Vector4 val)
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
