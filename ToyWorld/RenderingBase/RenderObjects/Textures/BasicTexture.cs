using OpenTK.Graphics.OpenGL;

namespace RenderingBase.RenderObjects.Textures
{
    public class BasicTexture : TextureBase
    {
        public BasicTexture(int width, int height, TextureTarget textureTarget = TextureTarget.Texture2D)
            : base(width, height, textureTarget)
        { }


        public new void Init(
            int[] data,
            PixelFormat dataFormat = PixelFormat.Bgra,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba,
            bool generateMipmap = false)
        {
            base.Init(data, dataFormat, internalDataFormat, generateMipmap);
        }

        public new void SetParameters(
            TextureMinFilter minFilter = TextureMinFilter.Linear,
            TextureMagFilter magFilter = TextureMagFilter.Linear,
            TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat)
        {
            base.SetParameters(minFilter, magFilter, wrapMode);
        }
    }
}
