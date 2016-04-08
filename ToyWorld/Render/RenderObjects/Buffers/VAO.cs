using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;

namespace Render.RenderObjects.Buffers
{
    internal class Vao : IDisposable
    {
        public enum VboPosition
        {
            Vertices = 0,
            TextureOffsets = 1,


            // Testing stuff
            TextureCoords = 6,
            Colors = 7,
        }



        public uint Handle { get; private set; }

        private readonly Dictionary<VboPosition, VboBase> m_vbos = new Dictionary<VboPosition, VboBase>();


        #region Genesis

        public Vao()
        {
            Handle = (uint)GL.GenVertexArray();
        }

        public void Dispose()
        {
            GL.DeleteVertexArray(Handle);

            foreach (VboBase vbo in m_vbos.Values)
                vbo.Dispose();

            m_vbos.Clear();
        }

        #endregion

        #region Indexing

        protected VboBase this[VboPosition id]
        {
            get
            {
                VboBase vbo;

                if (!m_vbos.TryGetValue(id, out vbo))
                    throw new ArgumentException("Access to not registered Vbo.", "id");

                return vbo;
            }
            set
            {
                if (m_vbos.ContainsKey(id))
                    throw new ArgumentException("A Vbo has already been registered to this id.", "id");

                m_vbos[id] = value;
            }
        }

        #endregion


        public void EnableAttrib(
            VboPosition id, int attribArrayIdx = -1,
            VertexAttribPointerType type = VertexAttribPointerType.Float,
            bool normalized = false,
            int stride = 0, int offset = 0)
        {
            if (attribArrayIdx < 0)
                attribArrayIdx = (int)id;


            VboBase vboBase = this[id];

            GL.BindVertexArray(Handle);
            vboBase.Bind();

            GL.EnableVertexAttribArray(attribArrayIdx);
            GL.VertexAttribPointer(attribArrayIdx, vboBase.ElementSize, type, normalized, stride, offset);

            //GL.BindVertexArray(0);
            //VboBase.Unbind();
        }

        public void DisableAttrib(string id, int attribArrayIdx)
        {
            GL.BindVertexArray(Handle);
            GL.DisableVertexAttribArray(attribArrayIdx);
        }
    }
}
