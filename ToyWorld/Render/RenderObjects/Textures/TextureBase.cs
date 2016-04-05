using System.Drawing;
using System.Drawing.Imaging;
using OpenTK.Graphics.OpenGL;
using PixelFormat = OpenTK.Graphics.OpenGL.PixelFormat;

namespace Render.RenderObjects.Textures
{
    internal class TextureBase
    {
        private readonly int m_handle;

        private readonly TextureTarget m_target;


        protected TextureBase(
            Bitmap bmp,
            TextureMinFilter minFilter = TextureMinFilter.Linear,
            TextureMagFilter magFilter = TextureMagFilter.Linear,
            TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat,
            TextureTarget textureTarget = TextureTarget.Texture2D)
        {
            m_target = textureTarget;
            m_handle = GL.GenTexture();
            GL.BindTexture(textureTarget, m_handle);

            GL.TexParameter(textureTarget, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(textureTarget, TextureParameterName.TextureMagFilter, (int)magFilter);
            GL.TexParameter(textureTarget, TextureParameterName.TextureWrapS, (int)wrapMode);
            GL.TexParameter(textureTarget, TextureParameterName.TextureWrapT, (int)wrapMode);

            var data = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);

            GL.TexImage2D(
                textureTarget,
                0,
                PixelInternalFormat.Rgba,
                bmp.Width, bmp.Height, 0,
                PixelFormat.Rgba, PixelType.UnsignedByte,
                data.Scan0);

            GL.BindTexture(textureTarget, 0);
        }


        public void Bind()
        {
            GL.BindTexture(m_target, m_handle);
        }
    }
}
