using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;

namespace Render.RenderObjects.Buffers
{
    internal class VAO : IDisposable
    {
        public uint Handle { get; private set; }

        private readonly Dictionary<string, VBOBase> m_vbos = new Dictionary<string, VBOBase>();


        #region Genesis

        public VAO()
        {
            Handle = (uint)GL.GenVertexArray();
        }

        public void Dispose()
        {
            GL.DeleteVertexArray(Handle);

            foreach (var vbo in m_vbos.Values)
                vbo.Dispose();

            m_vbos.Clear();
        }

        #endregion

        #region Indexing

        public VBOBase this[string id]
        {
            get
            {
                VBOBase vbo;

                if (!m_vbos.TryGetValue(id, out vbo))
                    throw new ArgumentException("Access to not registered VBOBase.", "id");

                return vbo;
            }
            set
            {
                if (m_vbos.ContainsKey(id))
                    throw new ArgumentException("A VBOBase has already been registered to this id.", "id");

                m_vbos[id] = value;
            }
        }

        #endregion


        public void EnableAttrib(
            string id, int attribArrayIdx,
            VertexAttribPointerType type = VertexAttribPointerType.Float,
            bool normalized = false,
            int stride = 0, int offset = 0)
        {
            VBOBase vboBase = this[id];

            GL.BindVertexArray(Handle);
            vboBase.Bind();

            GL.EnableVertexAttribArray(attribArrayIdx);
            GL.VertexAttribPointer(attribArrayIdx, vboBase.ElementSize, type, normalized, stride, offset);

            //GL.BindVertexArray(0);
            //VBOBase.Unbind();
        }

        public void DisableAttrib(string id, int attribArrayIdx)
        {
            GL.BindVertexArray(Handle);
            GL.DisableVertexAttribArray(attribArrayIdx);
        }
    }
}
