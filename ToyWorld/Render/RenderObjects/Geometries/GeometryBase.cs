using System;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal abstract class GeometryBase
    {
        protected readonly VAO Vao = new VAO();


        public abstract void Init();
        public abstract void Draw();


        #region Basic buffers

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

            return new VBO(buf.Length, buf, hint: BufferUsageHint.StaticDraw);
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

            return new VBO(buf.Length, buf, hint: BufferUsageHint.StaticDraw);
        }

        #endregion
    }
}
