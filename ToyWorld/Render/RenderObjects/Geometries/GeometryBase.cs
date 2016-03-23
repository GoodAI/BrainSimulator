using System;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal abstract class GeometryBase
    {
        protected VAO m_vao;
        public VBO IndexBuffer { get; set; }

        public abstract void Init();

        void Draw()
        {
        GL.BindVertexArray(m_vao.Handle);
        }


        #region Basic geometries

        protected static readonly Lazy<VBO> SquareVertices = new Lazy<VBO>(GenerateSquareVertices);
        static VBO GenerateSquareVertices()
        {
            float[] buf =
            {
                -1, 0, -1,
                1, 0, -1,
                1, 0, 1,
                -1, 0, 1
            };

            return new VBO(buf.Length, buf);
        }

        protected static readonly Lazy<VBO> SquareColors = new Lazy<VBO>(GenerateSquareColors);
        static VBO GenerateSquareColors()
        {
            float[] buf =
            {
                1, 0, 1,
                1, 1, 0,
                0, 1, 1,
                1, 1, 1
            };

            return new VBO(buf.Length, buf);
        }

        #endregion
    }
}
