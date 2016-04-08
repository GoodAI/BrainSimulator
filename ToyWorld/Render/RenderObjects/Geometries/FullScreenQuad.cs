using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    // 0 - Position
    internal class FullScreenQuad : GeometryBase
    {
        const string Vert = "vert";


        public FullScreenQuad()
        {
            this[Vert]= StaticVBOFactory.FullscreenQuadVertices.Value;
            EnableAttrib(Vert, 0);
        }


        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, 4);
        }
    }
}
