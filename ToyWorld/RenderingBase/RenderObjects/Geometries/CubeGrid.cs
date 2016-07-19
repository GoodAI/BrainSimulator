using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using VRageMath;

namespace RenderingBase.RenderObjects.Geometries
{
    public class CubeGrid : GeometryBase
    {
        public Vector2I Dimensions { get; private set; }


        public CubeGrid(Vector2I dimensions)
        {
            Dimensions = dimensions;

            this[VboPosition.Vertices] = StaticVboFactory.GetCubeGridVertices(dimensions);
            EnableAttrib(VboPosition.Vertices);

            this[OtherVbo.Elements] = StaticVboFactory.GetCubeGridElements(dimensions);
            GL.BindVertexArray(Handle);
            this[OtherVbo.Elements].Bind();
        }

        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawElements(PrimitiveType.Quads,  6 * 4, DrawElementsType.UnsignedShort, 0);
        }
    }
}
