using OpenTK.Graphics.OpenGL;

namespace RenderingBase.RenderObjects.Buffers
{
    public class Pbo<T>
        : Vbo<T>
        where T : struct
    {
        public Pbo()
            : base(target: BufferTarget.PixelPackBuffer)
        { }
    }
}
