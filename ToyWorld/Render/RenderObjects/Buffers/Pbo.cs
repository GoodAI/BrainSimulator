using OpenTK.Graphics.OpenGL;

namespace Render.RenderObjects.Buffers
{
    internal class Pbo : Vbo<uint>
    {
        public Pbo()
            : base(target: BufferTarget.PixelPackBuffer)
        { }
    }
}
