using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;

namespace Render.RenderObjects.Buffers
{
    internal class VAO : IDisposable
    {
        public uint Handle { get; private set; }

        protected readonly Dictionary<string, VBO> VBOs = new Dictionary<string, VBO>();


        #region Genesis

        public VAO()
        {
            Handle = (uint)GL.GenVertexArray();
        }

        public void Dispose()
        {
            GL.DeleteVertexArray(Handle);

            foreach (var vbo in VBOs.Values)
                vbo.Dispose();

            VBOs.Clear();
        }

        #endregion

        #region Indexing

        public VBO this[string id]
        {
            get
            {
                VBO vbo;

                if (!VBOs.TryGetValue(id, out vbo))
                    throw new ArgumentException("Access to not registered VBO.", "id");

                return vbo;
            }
            set
            {
                if (VBOs.ContainsKey(id))
                    throw new ArgumentException("A VBO has already been registered to this id.", "id");

                VBOs[id] = value;
            }
        }

        #endregion


        public void EnableAttrib(
            string id, int attribArrayIdx,
            VertexAttribPointerType type = VertexAttribPointerType.Float,
            bool normalized = false,
            int stride = 0, int offset = 0)
        {
            VBO vbo = this[id];

            GL.BindVertexArray(Handle);
            vbo.Bind();

            GL.EnableVertexAttribArray(attribArrayIdx);
            GL.VertexAttribPointer(attribArrayIdx, vbo.ElementSize, type, normalized, stride, offset);

            //GL.BindVertexArray(0);
            //vbo.Unbind();
        }

        public void DisableAttrib(string id, int attribArrayIdx)
        {
            GL.BindVertexArray(Handle);
            GL.DisableVertexAttribArray(attribArrayIdx);
        }
    }
}
