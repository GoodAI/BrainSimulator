using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace RenderingBase.RenderObjects.Effects
{
    public class EffectBase : IDisposable
    {
        protected const string ShaderPathBase = "RenderingBase.RenderObjects.Effects.Src.";

        private readonly int m_prog;

        private readonly Dictionary<int, int> m_uniformLocations = new Dictionary<int, int>();


        #region Genesis

        // TODO: genericity
        // Addenda serve as switchable extensions to base shaders -- can be used as a different implementation of functions defined in base shaders.
        protected EffectBase(Type uniformNamesEnumType, string vertPath, string fragPath, Stream vertAddendum = null, Stream fragAddendum = null)
        {
            // Load shaders
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


            // Setup uniform locations
            Debug.Assert(uniformNamesEnumType.IsEnum, "The passed type must be an enum type.");

            foreach (var value in Enum.GetValues(uniformNamesEnumType))
            {
                m_uniformLocations[Convert.ToInt32(value)] = GL.GetUniformLocation(m_prog, value.ToString());
            }
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


        #region Uniforms and indexing

        protected int this[Enum val]
        {
            get { return m_uniformLocations[Convert.ToInt32(val)]; }
        }


        protected void SetUniform1(int pos, int val)
        {
            GL.Uniform1(pos, val);
        }

        protected void SetUniform1(int pos, float val)
        {
            GL.Uniform1(pos, val);
        }

        protected void SetUniform2(int pos, Vector2I val)
        {
            GL.Uniform2(pos, val.X, val.Y);
        }

        protected void SetUniform2(int pos, Vector2 val)
        {
            GL.Uniform2(pos, val.X, val.Y);
        }

        protected void SetUniform3(int pos, Vector3I val)
        {
            GL.Uniform3(pos, val.X, val.Y, val.Z);
        }

        protected void SetUniform4(int pos, Vector4I val)
        {
            GL.Uniform4(pos, val.X, val.Y, val.Z, val.W);
        }

        protected void SetUniform4(int pos, Vector4 val)
        {
            GL.Uniform4(pos, val.X, val.Y, val.Z, val.W);
        }

        // TODO: Jde matrix prevest na array bez kopirovani a unsafe kodu?
        /// <summary>
        ///Passed matrices are applied from left to right (as in vert*(a*b*c) -- a will be first).
        /// </summary>
        protected void SetUniformMatrix4(int pos, Matrix val)
        {
            unsafe
            {
                GL.UniformMatrix4(pos, 1, false, val.M.data);
            }
        }

        #endregion
    }
}
