using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;
using Render.RenderObjects.Buffers;
using VRageMath;

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
            Vao[id].Update(data, count, offset);
        }

        #region Basic static buffers

        #region Square

        protected static readonly Lazy<VBO> FullscreenQuadVertices = new Lazy<VBO>(GenerateSquareVertices);
        private static VBO GenerateSquareVertices()
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
        private static VBO GenerateSquareColors()
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

        #region Grid

        private static readonly Dictionary<Vector2I, VBO> GridVertices = new Dictionary<Vector2I, VBO>();
        protected static VBO GetGridVertices(Vector2I gridSize)
        {
            {
                VBO grid;

                if (GridVertices.TryGetValue(gridSize, out grid))
                    return grid;
            }


            float[] vertices = new float[gridSize.Size()];

            int xStep = 1 / gridSize.X;
            int yStep = 1 / gridSize.Y;

            float x = 0, y = 0;

            for (int j = 0; j < gridSize.Y; j++)
            {
                for (int i = 0; i < gridSize.X; i++)
                {
                    int idx = i * j * 2;
                    vertices[idx] = x;
                    vertices[idx + 1] = y;

                    x += xStep;
                }

                y += yStep;
            }

            return new VBO(vertices.Length, vertices, 2, hint: BufferUsageHint.StaticDraw);
        }

        #endregion

        #endregion
    }
}
