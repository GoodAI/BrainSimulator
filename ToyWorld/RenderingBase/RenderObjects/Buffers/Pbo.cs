using OpenTK.Graphics.OpenGL;

namespace RenderingBase.RenderObjects.Buffers
{
    public class Pbo : Vbo<uint>
    {
        public Pbo()
            : base(target: BufferTarget.PixelPackBuffer)
        { }
    }
}
