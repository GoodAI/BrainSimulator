using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;

namespace RenderingBase.RenderObjects.Geometries
{
    public class FullScreenQuad : GeometryBase
    {
        public FullScreenQuad()
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
