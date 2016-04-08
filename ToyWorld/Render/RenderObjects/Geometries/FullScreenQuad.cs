using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal class FullScreenQuad : GeometryBase
    {
        public FullScreenQuad()
        {
            this[VboPosition.Vertices] = StaticVboFactory.FullscreenQuadVertices;
            EnableAttrib(VboPosition.Vertices);
        }


        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, 4);
        }
    }
}
