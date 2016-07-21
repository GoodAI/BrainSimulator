using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;

namespace RenderingBase.RenderObjects.Buffers
{
    public class Vao : IDisposable
    {
        public enum VboPosition
        {
            Vertices = 0,
            TextureOffsets = 1,


            // Testing stuff
            TextureCoords = 6,
            Colors = 7,
        }

        public enum OtherVbo // Not used as shader attributes
        {
            Elements = 1,
        }


        public uint Handle { get; private set; }

        private readonly Dictionary<int, VboBase> m_vbos = new Dictionary<int, VboBase>();


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
            get { return m_vbos[(int)id]; }
            set { m_vbos[(int)id] = value; }
        }

        const int KeyOffset = 0x0f000000;

        protected VboBase this[OtherVbo id]
        {
            get { return m_vbos[(int)id + KeyOffset]; }
            set { m_vbos[(int)id + KeyOffset] = value; }
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

        public void EnableAttribI(
            VboPosition id, int attribArrayIdx = -1,
            VertexAttribIntegerType type = VertexAttribIntegerType.Int,
            int stride = 0, int offset = 0)
        {
            if (attribArrayIdx < 0)
                attribArrayIdx = (int)id;


            VboBase vboBase = this[id];

            GL.BindVertexArray(Handle);
            vboBase.Bind();

            GL.EnableVertexAttribArray(attribArrayIdx);
            GL.VertexAttribIPointer(attribArrayIdx, vboBase.ElementSize, type, stride, new IntPtr(offset));

            //GL.BindVertexArray(0);
            //VboBase.Unbind();
        }

        //public void DisableAttrib(string id, int attribArrayIdx)
        //{
        //    GL.BindVertexArray(Handle);
        //    GL.DisableVertexAttribArray(attribArrayIdx);
        //}
    }
}
