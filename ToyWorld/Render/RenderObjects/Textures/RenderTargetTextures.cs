using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace Render.RenderObjects.Textures
{
    internal abstract class RenderTargetTexture : TextureBase
    {
        protected RenderTargetTexture(Vector2I size)
            : base(null,
                   size.X, size.Y,
                // Use default pixel format
                   minFilter: TextureMinFilter.Nearest,
                   magFilter: TextureMagFilter.Nearest,
                   wrapMode: TextureWrapMode.ClampToEdge)
        { }

        protected RenderTargetTexture(Vector2I size, PixelFormat pixelFormat, PixelInternalFormat internalFormat)
            : base(null,
                   size.X, size.Y,
                   pixelFormat,
                   internalFormat,
                   TextureMinFilter.Nearest,
                   TextureMagFilter.Nearest,
                   TextureWrapMode.ClampToEdge)
        { }
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
