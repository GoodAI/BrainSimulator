using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal abstract class GeometryBase : Vao
    {
        public abstract void Draw();

        public void Update<T>(VboPosition id, T[] data, int count = -1, int offset = 0)
            where T : struct
        {
            this[id].Update(data, count, offset);
        }
    }
}
