using System;
using System.IO;
using OpenTK.Graphics.OpenGL;
using VRageMath;
using PixelFormat = OpenTK.Graphics.OpenGL.PixelFormat;

namespace Render.RenderObjects.Textures
{
    internal class TextureBase : IDisposable
    {
        private readonly int m_handle;
        private readonly TextureTarget m_target;

        public int Handle { get { return m_handle; } }
        public TextureTarget Target { get { return m_target; } }
        public Vector2I Size { get; protected set; }


        protected TextureBase()
        { }

        public TextureBase(
            int[] data, int width, int height,
            PixelFormat dataFormat = PixelFormat.Bgra,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba,
            TextureMinFilter minFilter = TextureMinFilter.Linear,
            TextureMagFilter magFilter = TextureMagFilter.Linear,
            TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat,
            TextureTarget textureTarget = TextureTarget.Texture2D,
            bool generateMipmap = false)
        {
            Size = new Vector2I(width, height);

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
                internalDataFormat,
                width, height, 0,
                dataFormat, PixelType.UnsignedByte,
                data);

            if (generateMipmap)
                GL.GenerateMipmap((GenerateMipmapTarget)textureTarget);

            GL.BindTexture(textureTarget, 0);
        }

        public virtual void Dispose()
        {
            GL.DeleteTexture(m_handle);
        }


        public virtual void Bind()
        {
            GL.BindTexture(m_target, m_handle);
        }
    }
}
