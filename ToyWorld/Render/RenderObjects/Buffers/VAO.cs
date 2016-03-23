using System;
using System.Collections.Generic;
using System.Diagnostics;
using OpenTK.Graphics.OpenGL;

namespace Render.Geometries.Buffers
{
    internal class VAO : IDisposable
    {
        public uint Handle { get; private set; }

        private readonly List<VBO> m_BOs = new List<VBO>();


        public VAO()
        {
            Handle = (uint)GL.GenVertexArray();
        }

        public void Dispose()
        {
            GL.DeleteVertexArray(Handle);
        }

        public void AddVBO(VBO vbo, int index, VertexAttribPointerType type = VertexAttribPointerType.Float, bool normalized = false, int stride = 1, int offset = 0)
        {
            Debug.Assert(vbo != null);

            m_BOs.Add(vbo);

            GL.BindVertexArray(Handle);
            vbo.Bind();

            GL.EnableVertexAttribArray(index);
            GL.VertexAttribPointer(index, vbo.Count, type, normalized, stride, offset);

            GL.BindVertexArray(0);
            vbo.Unbind();
        }
    }
}
