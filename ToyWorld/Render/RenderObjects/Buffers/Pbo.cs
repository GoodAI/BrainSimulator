using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal class Pbo : Vbo<uint>
    {
        public Pbo(int tCount)
            : base(tCount, target: BufferTarget.PixelPackBuffer, hint: BufferUsageHint.StreamDraw)
        { }
    }
}
