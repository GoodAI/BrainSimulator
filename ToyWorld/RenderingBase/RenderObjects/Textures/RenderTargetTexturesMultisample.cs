using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace RenderingBase.RenderObjects.Textures
{
    internal abstract class RenderTargetTextureMultisample : TextureBase
    {
        protected RenderTargetTextureMultisample(Vector2I size, int sampleCount)
            : base(size.X, size.Y, TextureTarget.Texture2DMultisample) // Use default pixel format)
        {
            InitMultisample(sampleCount);
        }

        protected RenderTargetTextureMultisample(Vector2I size, int samepleCount, PixelInternalFormat internalFormat)
            : base(size.X, size.Y, TextureTarget.Texture2DMultisample)
        {
            InitMultisample(samepleCount, internalFormat);
        }
    }

    internal class RenderTargetColorTextureMultisample : RenderTargetTextureMultisample
    {
        public RenderTargetColorTextureMultisample(Vector2I size, int sampleCount)
            : base(size, sampleCount)
        { }
    }

    internal class RenderTargetDepthTextureMultisample : RenderTargetTextureMultisample
    {
        public RenderTargetDepthTextureMultisample(Vector2I size, int sampleCount)
            : base(size, sampleCount, PixelInternalFormat.DepthComponent32)
        { }
    }

    //internal class RenderTargetStencilTextureMultisample : RenderTargetTextureMultisample
    //{
    //    public RenderTargetStencilTextureMultisample(Vector2I size)
    //        : base(size, PixelFormat.StencilIndex)
    //    { }
    //}
}
