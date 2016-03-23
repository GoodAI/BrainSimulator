using Render.Geometries.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal class Square : GeometryBase
    {
        public override void Init()
        {
            m_vao.AddVBO(SquareVertices.Value, 0);
            m_vao.AddVBO(SquareColors.Value, 1);
        }
    }
}
