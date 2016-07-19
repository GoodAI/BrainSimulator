using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;

namespace RenderingBase.RenderObjects.Geometries
{
    public class Cube : GeometryBase
    {
        public Cube()
        {
            this[VboPosition.Vertices] = StaticVboFactory.CubeVertices;
            EnableAttrib(VboPosition.Vertices);

            this[OtherVbo.Elements] = StaticVboFactory.CubeElements;
            GL.BindVertexArray(Handle);
            this[OtherVbo.Elements].Bind();
        }


        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawElements(PrimitiveType.Quads, 6 * 4, DrawElementsType.UnsignedShort, 0);
        }
    }
}
