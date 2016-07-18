using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;

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
            GL.DrawElements(PrimitiveType.Quads, 6, DrawElementsType.UnsignedByte, 0);
        }
    }
}
