using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;

namespace RenderingBase.RenderObjects.Geometries
{
    public class Quad : GeometryBase
    {
        public Quad()
        {
            this[VboPosition.Vertices] = StaticVboFactory.QuadVertices;
            EnableAttrib(VboPosition.Vertices);
        }


        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, 4);
        }
    }
}
