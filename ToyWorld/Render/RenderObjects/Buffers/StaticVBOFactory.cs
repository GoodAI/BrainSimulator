using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal static class StaticVBOFactory
    {
        private static readonly Dictionary<Vector2I, VBOBase> GridVertices = new Dictionary<Vector2I, VBOBase>();

        public static VBOBase GetGridVertices(Vector2I gridSize) { return GetGridVerticesInternal(gridSize); }
        public static Lazy<VBOBase> FullscreenQuadVertices;
        public static Lazy<VBOBase> QuadColors;


        public static void Init()
        {
            FullscreenQuadVertices = new Lazy<VBOBase>(GenerateSquareVertices);
            QuadColors = new Lazy<VBOBase>(GenerateSquareColors);
        }

        public static void Clear()
        {
            GridVertices.Clear();

            FullscreenQuadVertices = null;
            QuadColors = null;
        }


        #region Basic static buffers

        #region Square

        private static VBOBase GenerateSquareVertices()
        {
            float[] squareVertices =
            {
                -1,-1,
                 1,-1,
                 1, 1,
                -1, 1,
            };

            return new VBO<float>(squareVertices.Length, squareVertices, 2, hint: BufferUsageHint.StaticDraw);
        }

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

        private static VBOBase GetGridVerticesInternal(Vector2I gridSize)
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
