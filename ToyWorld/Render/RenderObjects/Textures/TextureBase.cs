using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;
using PixelFormat = OpenTK.Graphics.OpenGL.PixelFormat;

namespace Render.RenderObjects.Textures
{
    internal class TextureBase
    {
        private readonly int m_handle;

        private readonly TextureTarget m_target;


        public TextureBase()
        { }

        public TextureBase(
            MemoryStream data, int width, int height,
            PixelFormat streamDataFormat = PixelFormat.Rgba,
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

            GL.TexImage2D(
                textureTarget,
                0,
                PixelInternalFormat.Rgba,
                width, height, 0,
                streamDataFormat, PixelType.UnsignedByte,
                data.ToArray());

            GL.BindTexture(textureTarget, 0);
        }


        public virtual void Bind()
        {
            GL.BindTexture(m_target, m_handle);
        }
    }
}
