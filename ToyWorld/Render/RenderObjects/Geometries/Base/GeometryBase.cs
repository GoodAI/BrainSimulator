using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal abstract class GeometryBase : Vao
    {
        public abstract void Draw();

        protected void Update<T>(VboPosition id, T[] data, int count = -1, int offset = 0)
            where T : struct
        {
            // No cast checking, should be covered by tests and usage
            ((Vbo<T>)base[id]).Update(data, count, offset);
        }
    }
}
