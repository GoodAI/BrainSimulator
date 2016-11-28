using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace RenderingBase.RenderObjects.Textures
{
    public class BasicTexture1D : TextureBase
    {
        public BasicTexture1D(Vector2I size)
            : this(size.Size())
        { }

        public BasicTexture1D(int width, TextureTarget textureTarget = TextureTarget.Texture1D)
            : base(width, 1, textureTarget)
        { }

        public void DefaultInit()
        {
            Init1D(PixelFormat.RedInteger, PixelInternalFormat.R16ui, PixelType.UnsignedShort);
            SetParameters(TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.ClampToBorder);
        }
    }

    public class BasicTexture2D : TextureBase
    {
        public BasicTexture2D(Vector2I size)
            : this(size.X, size.Y)
        { }

        public BasicTexture2D(int width, int height, TextureTarget textureTarget = TextureTarget.Texture2D)
            : base(width, height, textureTarget)
        { }

        public void DefaultInit()
        {
            Init2D(PixelFormat.RedInteger, PixelInternalFormat.R16ui, PixelType.UnsignedShort);
            SetParameters(TextureMinFilter.Nearest, TextureMagFilter.Nearest, TextureWrapMode.ClampToBorder);
        }
    }
}
