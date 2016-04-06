using System;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;
using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal abstract class GeometryBase : IDisposable
    {
        protected readonly VAO Vao = new VAO();


        public void Dispose()
        {
            Vao.Dispose();
        }


        public abstract void Draw();

        public void Update<T>(string id, T[] data, int count = -1, int offset = 0)
            where T : struct
        {
            Vao.VBOs[id].Update(data, count, offset);
        }

        #region Basic buffers

        protected static readonly Lazy<VBO> FullscreenQuadVertices = new Lazy<VBO>(GenerateSquareVertices);
        static VBO GenerateSquareVertices()
        {
            float[] buf =
            {
                -1,-1, 0, 
                 1,-1, 0, 
                 1, 1, 0, 
                -1, 1, 0, 
            };

            return new VBO(buf.Length, buf, 3, hint: BufferUsageHint.StaticDraw);
        }

        protected static readonly Lazy<VBO> QuadColors = new Lazy<VBO>(GenerateSquareColors);
        static VBO GenerateSquareColors()
        {
            float[] buf =
            {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                1, 1, 1,
            };

            return new VBO(buf.Length, buf, 3, hint: BufferUsageHint.StaticDraw);
        }

        #endregion
    }
}
