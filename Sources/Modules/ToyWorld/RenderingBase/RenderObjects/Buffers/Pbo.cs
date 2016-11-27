using OpenTK.Graphics.OpenGL;

namespace RenderingBase.RenderObjects.Buffers
{
    public class Pbo<T>
        : Vbo<T>
        where T : struct
    {
        public Pbo(int elementSize = -1)
            : base(target: BufferTarget.PixelPackBuffer, elementSize: elementSize)
        { }
    }
}
