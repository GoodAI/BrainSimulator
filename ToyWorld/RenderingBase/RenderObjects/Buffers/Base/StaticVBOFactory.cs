using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace RenderingBase.RenderObjects.Buffers
{
    internal static class StaticVboFactory
    {
        private static readonly Dictionary<Vector2I, VboBase> GridVertices = new Dictionary<Vector2I, VboBase>();


        #region Public getters

        public static VboBase QuadVertices { get { return _quadVertices.Value; } }
        public static VboBase CubeVertices { get { return _cubeVertices.Value; } }
        public static VboBase CubeElements { get { return _cubeElements.Value; } }
        public static VboBase QuadColors { get { return _quadColors.Value; } }

        public static VboBase GetGridVertices(Vector2I gridSize) { return GetGridVerticesInternal(gridSize); }

        #endregion

        #region Genesis

        public static void Init()
        {
            _quadVertices = new Lazy<VboBase>(GenerateSquareVertices);
            _cubeVertices = new Lazy<VboBase>(GenerateCubeVertices);
            _cubeElements = new Lazy<VboBase>(GenerateCubeElements);
            _quadColors = new Lazy<VboBase>(GenerateSquareColors);
        }

        public static void Clear()
        {
            GridVertices.Clear();

            _quadVertices = null;
            _quadColors = null;
        }

        #endregion

        #region Basic static buffers

        #region Square

        private static Lazy<VboBase> _quadVertices;
        private static VboBase GenerateSquareVertices()
        {
            Vector2[] squareVertices =
            {
               new Vector2(-1,-1),
               new Vector2(-1, 1),
               new Vector2( 1, 1),
               new Vector2( 1,-1),
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

        #region Cube

        private static Lazy<VboBase> _cubeVertices;
        private static VboBase GenerateCubeVertices()
        {
            Vector3[] cubeVertices =
            {
               new Vector3(-1,-1,-1),
               new Vector3(-1, 1,-1),
               new Vector3( 1, 1,-1),
               new Vector3( 1,-1,-1),
               new Vector3(-1,-1, 1),
               new Vector3(-1, 1, 1),
               new Vector3( 1, 1, 1),
               new Vector3( 1,-1, 1),
            };

            return new StaticVbo<Vector3>(cubeVertices.Length, cubeVertices, 3, hint: BufferUsageHint.StaticDraw);
        }

        private static Lazy<VboBase> _cubeElements;
        private static VboBase GenerateCubeElements()
        {
            Vector4UByte[] cubeElements =
            {
               new Vector4UByte(3, 2, 1, 0), // Back
               new Vector4UByte(4, 5, 6, 7), // Front
               new Vector4UByte(4, 5, 1, 0), // Left
               new Vector4UByte(7, 6, 2, 3), // Right
               new Vector4UByte(5, 1, 2, 6), // Up
               new Vector4UByte(7, 3, 0, 4), // Down
            };

            return new StaticVbo<Vector4UByte>(cubeElements.Length, cubeElements, 1, hint: BufferUsageHint.StaticDraw, target: BufferTarget.ElementArrayBuffer);
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
            Vector2I yStep = new Vector2I(0, 2);
            Vector2I xyStep = xStep + yStep;
            Vector2 gridSizeInv = 1 / new Vector2(gridSize.X, gridSize.Y);

            // Generate tiles from bot-left corner row-wise, centered on origin
            Vector2I botLeft = new Vector2I(-gridSize.X, -gridSize.Y);

            int idx = 0;

            for (int j = 0; j < gridSize.Y; j++)
            {
                for (int i = 0; i < gridSize.X; i++)
                {
                    // Start top-left, continue clock-wise
                    vertices[idx++] = (Vector2)botLeft * gridSizeInv;
                    vertices[idx++] = (Vector2)(botLeft + yStep) * gridSizeInv;
                    vertices[idx++] = (Vector2)(botLeft + xyStep) * gridSizeInv;
                    vertices[idx++] = (Vector2)(botLeft + xStep) * gridSizeInv;

                    botLeft += xStep;
                }

                botLeft += yStep;
                botLeft.X = -gridSize.X;
            }

            return new StaticVbo<Vector2>(vertices.Length, vertices, 2, hint: BufferUsageHint.StaticDraw);
        }

        #endregion

        #endregion
    }
}
