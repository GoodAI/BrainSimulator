using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;

namespace Render.RenderObjects.Geometries
{
    // 0 - Position
    internal class FullScreenQuad : GeometryBase
    {
        public FullScreenQuad()
        {
            Vao.VBOs.Add("vert", FullscreenQuadVertices.Value);
            Vao.EnableVBO("vert", 0);
        }


        public override void Draw()
        {
            GL.BindVertexArray(Vao.Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, 4);
        }
    }
}
