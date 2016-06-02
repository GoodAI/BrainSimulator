using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using VRageMath;

namespace RenderingBase.RenderObjects.Geometries
{
    public class FullScreenGrid : GeometryBase
    {
        public Vector2I Dimensions { get; private set; }


        public FullScreenGrid(Vector2I dimensions)
        {
            Dimensions = dimensions;

            this[VboPosition.Vertices] = StaticVboFactory.GetGridVertices(dimensions);
            EnableAttrib(VboPosition.Vertices);
        }

        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, Dimensions.Size() * 4);
        }
    }
}
