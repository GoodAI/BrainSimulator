using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Geometries;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal static class StaticVboFactory
    {
        private static readonly Dictionary<Vector2I, VboBase> GridVertices = new Dictionary<Vector2I, VboBase>();


        #region Public getters

        public static VboBase FullscreenQuadVertices { get { return _fullscreenQuadVertices.Value; } }
        public static VboBase QuadColors { get { return _quadColors.Value; } }

        public static VboBase GetGridVertices(Vector2I gridSize) { return GetGridVerticesInternal(gridSize); }

        #endregion

        #region Genesis

        public static void Init()
        {
            _fullscreenQuadVertices = new Lazy<VboBase>(GenerateSquareVertices);
            _quadColors = new Lazy<VboBase>(GenerateSquareColors);
        }

        public static void Clear()
        {
            GridVertices.Clear();

            _fullscreenQuadVertices = null;
            _quadColors = null;
        }

        #endregion

        #region Basic static buffers

        #region Square

        private static Lazy<VboBase> _fullscreenQuadVertices;
        private static VboBase GenerateSquareVertices()
        {
            Vector2[] squareVertices =
            {
               new Vector2(-1, 1),
               new Vector2( 1, 1),
               new Vector2( 1,-1),
               new Vector2(-1,-1),
            };

            return new StaticVbo<Vector2>(squareVertices.Length, squareVertices, 2, hint: BufferUsageHint.StaticDraw);
        }

        private static Lazy<VboBase> _quadColors;
        private static VboBase GenerateSquareColors()
        {
            float[] buf =
            {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                1, 1, 1,
            };

            return new StaticVbo<float>(buf.Length, buf, 3, hint: BufferUsageHint.StaticDraw);
        }

        #endregion

        #region Grid

        private static VboBase GetGridVerticesInternal(Vector2I gridSize)
        {
            {
                VboBase grid;

                if (GridVertices.TryGetValue(gridSize, out grid))
                    return grid;
            }


            Vector2[] vertices = new Vector2[gridSize.Size() * 4];

            Vector2I xStep = new Vector2I(2, 0);
            Vector2I yStep = new Vector2I(0, -2);
            Vector2I xyStep = xStep + yStep;
            Vector2 gridSizeInv = 1 / new Vector2(gridSize.X, gridSize.Y);

            // Generate tiles from top-left corner row-wise, centered on origin
            Vector2I topLeft = new Vector2I(-gridSize.X, gridSize.Y);

            int idx = 0;

            for (int j = 0; j < gridSize.Y; j++)
            {
                for (int i = 0; i < gridSize.X; i++)
                {
                    // Start top-left, continue clock-wise
                    vertices[idx++] = (Vector2)topLeft * gridSizeInv;
                    vertices[idx++] = (Vector2)(topLeft + xStep) * gridSizeInv;
                    vertices[idx++] = (Vector2)(topLeft + xyStep) * gridSizeInv;
                    vertices[idx++] = (Vector2)(topLeft + yStep) * gridSizeInv;

                    topLeft += xStep;
                }

                topLeft += yStep;
                topLeft.X = -gridSize.X;
            }

            return new StaticVbo<Vector2>(vertices.Length, vertices, 2, hint: BufferUsageHint.StaticDraw);
        }

        #endregion

        #endregion
    }
}
