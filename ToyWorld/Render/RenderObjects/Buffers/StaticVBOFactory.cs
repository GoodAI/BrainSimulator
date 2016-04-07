using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal static class StaticVBOFactory
    {
        #region Basic static buffers

        #region Square

        static readonly float[] SquareVertices =
            {
                -1,-1,
                 1,-1,
                 1, 1,
                -1, 1,
            };

        public static readonly Lazy<VBOBase> FullscreenQuadVertices = new Lazy<VBOBase>(GenerateSquareVertices);
        private static VBOBase GenerateSquareVertices()
        {

            return new VBO<float>(SquareVertices.Length, SquareVertices, 2, hint: BufferUsageHint.StaticDraw);
        }

        public static readonly Lazy<VBOBase> QuadColors = new Lazy<VBOBase>(GenerateSquareColors);
        private static VBOBase GenerateSquareColors()
        {
            float[] buf =
            {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                1, 1, 1,
            };

            return new VBO<float>(buf.Length, buf, 3, hint: BufferUsageHint.StaticDraw);
        }

        #endregion

        #region Grid

        private static readonly Dictionary<Vector2I, VBOBase> GridVertices = new Dictionary<Vector2I, VBOBase>();

        public static VBOBase GetGridVertices(Vector2I gridSize)
        {
            {
                VBOBase grid;

                if (GridVertices.TryGetValue(gridSize, out grid))
                    return grid;
            }


            Vector2[] vertices = new Vector2[gridSize.Size() * 4];

            Vector2 xStep = new Vector2(1f / gridSize.X, 0);
            Vector2 yStep = new Vector2(0, 1f / gridSize.Y);

            Vector2 bl = new Vector2();

            int idx = 0;

            for (int j = 0; j < gridSize.Y; j++)
            {
                for (int i = 0; i < gridSize.X; i++)
                {
                    vertices[idx++] = bl;
                    vertices[idx++] = bl + xStep;
                    vertices[idx++] = bl + yStep;
                    vertices[idx++] = bl + xStep + yStep;

                    bl += xStep;
                }

                bl += yStep;
            }

            return new VBO<Vector2>(vertices.Length, vertices, 2, hint: BufferUsageHint.StaticDraw);
        }

        #endregion

        #endregion
    }
}
