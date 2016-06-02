using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace RenderingBase.RenderObjects.Textures
{
    internal abstract class RenderTargetTexture : TextureBase
    {
        protected RenderTargetTexture(Vector2I size)
            : base(size.X, size.Y) // Use default pixel format)
        {
            Init(null);
            SetParameters(
                   TextureMinFilter.Nearest,
                   TextureMagFilter.Nearest,
                   TextureWrapMode.ClampToEdge);
        }

        protected RenderTargetTexture(Vector2I size, PixelFormat pixelFormat, PixelInternalFormat internalFormat)
            : base(size.X, size.Y)
        {
            Init(null, pixelFormat, internalFormat);
            SetParameters(
                   TextureMinFilter.Nearest,
                   TextureMagFilter.Nearest,
                   TextureWrapMode.ClampToEdge);
        }
    }

    internal class RenderTargetColorTexture : RenderTargetTexture
    {
        public RenderTargetColorTexture(Vector2I size)
            : base(size)
        { }
    }

    internal class RenderTargetDepthTexture : RenderTargetTexture
    {
        public RenderTargetDepthTexture(Vector2I size)
            : base(size, PixelFormat.DepthComponent, PixelInternalFormat.DepthComponent)
        { }
    }

    //internal class RenderTargetStencilTexture : RenderTargetTexture
    //{
    //    public RenderTargetStencilTexture(Vector2I size)
    //        : base(size, PixelFormat.StencilIndex)
    //    { }
    //}
}
