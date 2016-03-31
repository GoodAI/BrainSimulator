using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal class Square : GeometryBase
    {
        public override void Init()
        {
            Vao.AddVBO(SquareVertices.Value, 0);
            Vao.AddVBO(SquareColors.Value, 1);
        }

        public override void Draw()
        {
            GL.BindVertexArray(Vao.Handle);
            // GL.DrawArrays(PrimitiveType.Quads, 0, 4);
            GL.BindVertexArray(0);
        }
    }
}
