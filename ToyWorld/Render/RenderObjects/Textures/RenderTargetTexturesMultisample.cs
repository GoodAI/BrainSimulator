using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace Render.RenderObjects.Textures
{
    internal abstract class RenderTargetTextureMultisample : TextureBase
    {
        protected RenderTargetTextureMultisample(Vector2I size, int sampleCount)
            : base(size.X, size.Y) // Use default pixel format)
        {
            InitMultisample(sampleCount);
            SetParameters(
                   TextureMinFilter.Nearest,
                   TextureMagFilter.Nearest,
                   TextureWrapMode.ClampToEdge);
        }

        protected RenderTargetTextureMultisample(Vector2I size, int samepleCount, PixelInternalFormat internalFormat)
            : base(size.X, size.Y)
        {
            InitMultisample(samepleCount, internalFormat);
            SetParameters(
                   TextureMinFilter.Nearest,
                   TextureMagFilter.Nearest,
                   TextureWrapMode.ClampToEdge);
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
            : base(size, sampleCount, PixelInternalFormat.DepthComponent)
        { }
    }

    //internal class RenderTargetStencilTextureMultisample : RenderTargetTexture
    //{
    //    public RenderTargetStencilTextureMultisample(Vector2I size)
    //        : base(size, PixelFormat.StencilIndex)
    //    { }
    //}
}
