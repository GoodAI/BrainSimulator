using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using VRageMath;
using VRageMath.PackedVector;

namespace RenderingBase.RenderObjects.Buffers
{
    internal static class StaticVboFactory
    {
        private static readonly Dictionary<Vector2I, Tuple<Vector2[], VboBase>> DuplicatedGridVertices = new Dictionary<Vector2I, Tuple<Vector2[], VboBase>>();
        private static readonly Dictionary<Vector2I, Tuple<Vector3[], VboBase>> CubeGridVertices = new Dictionary<Vector2I, Tuple<Vector3[], VboBase>>();
        private static readonly Dictionary<Vector2I, Tuple<HalfVector4[], VboBase>> CubeGridElements = new Dictionary<Vector2I, Tuple<HalfVector4[], VboBase>>();
        private static readonly Dictionary<Vector2I, Tuple<Vector3[], VboBase>> DuplicatedCubeGridVertices = new Dictionary<Vector2I, Tuple<Vector3[], VboBase>>();


        #region Public getters

        public static VboBase QuadVertices { get { return _quadVertices.Value; } }
        public static VboBase CubeVertices { get { return _cubeVertices.Value; } }
        public static VboBase CubeElements { get { return _cubeElements.Value; } }
        public static VboBase DuplicatedCubeVertices { get { return _duplicatedCubeVertices.Value; } }
        public static VboBase QuadColors { get { return _quadColors.Value; } }

        public static VboBase GetDuplicatedGridVertices(Vector2I gridSize) { return GenerateDuplicatedGridVertices(gridSize); }
        public static VboBase GetCubeGridVertices(Vector2I gridSize) { return GenerateCubeGridVertices(gridSize); }
        public static VboBase GetCubeGridElements(Vector2I gridSize) { return GenerateCubeGridElements(gridSize); }
        public static VboBase GetDuplicatedCubeGridVertices(Vector2I gridSize) { return GenerateDuplicatedCubeGridVertices(gridSize); }

        #endregion

        #region Genesis

        public static void Init()
        { }

        public static void Clear()
        {
            _quadVertices = null;
            _quadColors = null;

            _cubeVertices = null;
            _cubeElements = null;
            _duplicatedCubeVertices = null;

            DuplicatedGridVertices.Clear();
            CubeGridVertices.Clear();
            CubeGridElements.Clear();
        }

        #endregion

        #region Basic static buffers

        #region Square

        private static Lazy<VboBase> _quadVertices = new Lazy<VboBase>(GenerateSquareVertices);
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

        private static Lazy<VboBase> _quadColors = new Lazy<VboBase>(GenerateSquareColors);
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

        private static Lazy<VboBase> _cubeVertices = new Lazy<VboBase>(GenerateCubeVertices);
        private static Lazy<VboBase> _cubeElements = new Lazy<VboBase>(GenerateCubeElements);
        private static Lazy<VboBase> _duplicatedCubeVertices = new Lazy<VboBase>(GenerateDuplicatedCubeVertices);


        private static Vector3[] GenerateRawCubeVertices()
        {
            return new[]
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
        }

        private static VboBase GenerateCubeVertices()
        {
            Vector3[] cubeVertices = GenerateRawCubeVertices();
            return new StaticVbo<Vector3>(cubeVertices.Length, cubeVertices, 3, hint: BufferUsageHint.StaticDraw);
        }


        private static HalfVector4[] GenerateRawCubeElements()
        {
            return new[]
            {
               new HalfVector4(3, 2, 1, 0), // Back
               new HalfVector4(4, 5, 6, 7), // Front
               new HalfVector4(1, 5, 4, 0), // Left
               new HalfVector4(3, 7, 6, 2), // Right
               new HalfVector4(2, 6, 5, 1), // Up
               new HalfVector4(0, 4, 7, 3), // Down
            };
        }

        private static VboBase GenerateCubeElements()
        {
            HalfVector4[] cubeElements = GenerateRawCubeElements();
            return new StaticVbo<HalfVector4>(cubeElements.Length, cubeElements, 1, hint: BufferUsageHint.StaticDraw, target: BufferTarget.ElementArrayBuffer);
        }


        private static VboBase GenerateDuplicatedCubeVertices()
        {
            Vector3[] cubeVertices = new Vector3[6 * 4];
            Vector3[] rawCubeVertices = GenerateRawCubeVertices();
            HalfVector4[] rawCubeElements = GenerateRawCubeElements();

            int idx = 0;

            for (int i = 0; i < rawCubeElements.Length; i++)
            {
                cubeVertices[idx++] = rawCubeVertices[rawCubeElements[i].X];
                cubeVertices[idx++] = rawCubeVertices[rawCubeElements[i].Y];
                cubeVertices[idx++] = rawCubeVertices[rawCubeElements[i].Z];
                cubeVertices[idx++] = rawCubeVertices[rawCubeElements[i].W];
            }

            return new StaticVbo<Vector3>(cubeVertices.Length, cubeVertices, 3, hint: BufferUsageHint.StaticDraw);
        }

        #endregion

        #region Grid

        #region Helpers

        private static TFirst FirstFactoryHelper<TFirst, TSecond, TKey>(
            Func<TFirst> firstInitializer,
            Dictionary<TKey, Tuple<TFirst, TSecond>> dict,
            TKey key)
            where TSecond : class
        {
            Tuple<TFirst, TSecond> res;

            if (dict.TryGetValue(key, out res))
                return res.Item1;

            TFirst val = firstInitializer();

            res = new Tuple<TFirst, TSecond>(val, null);
            dict[key] = res;
            return res.Item1;
        }

        private static TSecond SecondFactoryHelper<TFirst, TSecond, TKey>(
            Func<TFirst> firstInitializer,
            Func<TFirst, TSecond> secondInitializer,
            Dictionary<TKey, Tuple<TFirst, TSecond>> dict,
            TKey key)
            where TSecond : class
        {
            Tuple<TFirst, TSecond> res;
            TFirst first;

            if (dict.TryGetValue(key, out res))
            {
                if (res.Item2 == null)
                {
                    first = res.Item1;
                    goto addToCache;
                }

                return res.Item2;
            }


            first = firstInitializer();

        addToCache:
            TSecond second = secondInitializer(first);
            res = new Tuple<TFirst, TSecond>(first, second);
            dict[key] = res;
            return res.Item2;

        }

        #endregion

        #region Quad

        private static Vector2[] GenerateRawDuplicatedGridVertices(Vector2I gridSize)
        {
            Func<Vector2[]> initializer = () =>
            {
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

                return vertices;
            };

            return FirstFactoryHelper(initializer, DuplicatedGridVertices, gridSize);
        }

        private static VboBase GenerateDuplicatedGridVertices(Vector2I gridSize)
        {
            Func<Vector2[]> firstInitializer = () => GenerateRawDuplicatedGridVertices(gridSize);
            Func<Vector2[], StaticVbo<Vector2>> secondInitializer = first => new StaticVbo<Vector2>(first.Length, first, 2, hint: BufferUsageHint.StaticDraw);

            return SecondFactoryHelper(firstInitializer, secondInitializer, DuplicatedGridVertices, gridSize);
        }

        #endregion

        #region Cube

        private static Vector3[] GenerateRawCubeGridVertices(Vector2I gridSize)
        {
            Func<Vector3[]> initializer = () =>
            {
                Vector3[] vertices = new Vector3[gridSize.Size() * 4 * 2];
                Vector2[] rawGridVertices = GenerateRawDuplicatedGridVertices(gridSize);

                Vector3 zStep = new Vector3(0, 0, 2);
                int idx = 0;

                for (int j = 0; j < gridSize.Y; j++)
                {
                    for (int i = 0; i < gridSize.X; i++)
                    {
                        // Bottom face
                        int startIdx = idx >> 1;

                        for (int k = startIdx; k < startIdx + 4; k++)
                            vertices[idx++] = new Vector3(rawGridVertices[k]);

                        // Top face
                        for (int k = 0; k < 4; k++)
                            vertices[idx++] = vertices[idx - 5] + zStep;
                    }
                }

                return vertices;
            };

            return FirstFactoryHelper(initializer, CubeGridVertices, gridSize);
        }

        private static VboBase GenerateCubeGridVertices(Vector2I gridSize)
        {
            Func<Vector3[]> firstInitializer = () => GenerateRawCubeGridVertices(gridSize);
            Func<Vector3[], StaticVbo<Vector3>> secondInitializer = first => new StaticVbo<Vector3>(first.Length, first, 3, hint: BufferUsageHint.StaticDraw);

            return SecondFactoryHelper(firstInitializer, secondInitializer, CubeGridVertices, gridSize);
        }

        private static HalfVector4[] GenerateRawCubeGridElements(Vector2I gridSize)
        {
            Func<HalfVector4[]> initializer = () =>
            {
                HalfVector4[] cubeElements = GenerateRawCubeElements();
                HalfVector4[] gridElements = new HalfVector4[gridSize.Size() * cubeElements.Length];

                for (int i = 0; i < gridSize.Size(); i++)
                {
                    int baseIdx = i * cubeElements.Length;
                    ushort offset = (ushort)(i * 8);

                    for (int j = 0; j < cubeElements.Length; j++)
                        gridElements[baseIdx + j] = cubeElements[j] + offset;
                }

                return gridElements;
            };

            return FirstFactoryHelper(initializer, CubeGridElements, gridSize);
        }

        private static VboBase GenerateCubeGridElements(Vector2I gridSize)
        {
            Func<HalfVector4[]> firstInitializer = () => GenerateRawCubeGridElements(gridSize);
            Func<HalfVector4[], StaticVbo<HalfVector4>> secondInitializer = first => new StaticVbo<HalfVector4>(first.Length, first, 1, hint: BufferUsageHint.StaticDraw, target: BufferTarget.ElementArrayBuffer);

            return SecondFactoryHelper(firstInitializer, secondInitializer, CubeGridElements, gridSize);
        }

        #endregion

        #region Duplicated Cube

        private static Vector3[] GenerateRawDuplicatedCubeGridVertices(Vector2I gridSize)
        {
            Func<Vector3[]> initializer = () =>
            {
                Vector3[] vertices = new Vector3[gridSize.Size() * 4 * 6];
                Vector3[] rawCubeGridVertices = GenerateRawCubeGridVertices(gridSize);
                HalfVector4[] rawCubeGridElements = GenerateRawCubeGridElements(gridSize);

                int idx = 0;

                for (int i = 0; i < rawCubeGridElements.Length; i++)
                {
                    vertices[idx++] = rawCubeGridVertices[rawCubeGridElements[i].X];
                    vertices[idx++] = rawCubeGridVertices[rawCubeGridElements[i].Y];
                    vertices[idx++] = rawCubeGridVertices[rawCubeGridElements[i].Z];
                    vertices[idx++] = rawCubeGridVertices[rawCubeGridElements[i].W];
                }

                return vertices;
            };

            return FirstFactoryHelper(initializer, DuplicatedCubeGridVertices, gridSize);
        }

        private static VboBase GenerateDuplicatedCubeGridVertices(Vector2I gridSize)
        {
            Func<Vector3[]> firstInitializer = () => GenerateRawDuplicatedCubeGridVertices(gridSize);
            Func<Vector3[], StaticVbo<Vector3>> secondInitializer = first => new StaticVbo<Vector3>(first.Length, first, 3, hint: BufferUsageHint.StaticDraw);

            return SecondFactoryHelper(firstInitializer, secondInitializer, DuplicatedCubeGridVertices, gridSize);
        }

        #endregion

        #endregion

        #endregion
    }
}
