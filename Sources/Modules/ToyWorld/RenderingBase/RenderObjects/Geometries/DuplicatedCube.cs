using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;

namespace RenderingBase.RenderObjects.Geometries
{
    public class DuplicatedCube : GeometryBase
    {
        public DuplicatedCube()
        {
            this[VboPosition.Vertices] = StaticVboFactory.DuplicatedCubeVertices;
            EnableAttrib(VboPosition.Vertices);
        }


        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, 6 * 4);
        }
    }
}
