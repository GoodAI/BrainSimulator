using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using VRageMath;

namespace RenderingBase.RenderObjects.Geometries
{
    public class DuplicatedCubeGrid : GeometryBase
    {
        public const int FaceCount = 6;
        public int PrimitiveFaceCount { get { return 6; } }

        public Vector2I Dimensions { get; private set; }


        public DuplicatedCubeGrid(Vector2I dimensions)
        {
            Dimensions = dimensions;

            this[VboPosition.Vertices] = StaticVboFactory.GetDuplicatedCubeGridVertices(dimensions);
            EnableAttrib(VboPosition.Vertices);
        }

        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, Dimensions.Size() * FaceCount * 4);
        }
    }
}
