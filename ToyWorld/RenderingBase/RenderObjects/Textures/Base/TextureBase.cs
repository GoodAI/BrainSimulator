using System;
using OpenTK.Graphics.OpenGL;
using VRageMath;

namespace RenderingBase.RenderObjects.Textures
{
    public abstract class TextureBase : IDisposable
    {
        private readonly int m_handle;
        private readonly TextureTarget m_target;
        private PixelFormat m_internalFormat;

        public int Handle { get { return m_handle; } }
        public TextureTarget Target { get { return m_target; } }
        public Vector2I Size { get; protected set; }


        protected TextureBase()
        { }

        protected TextureBase(int width, int height, TextureTarget target = TextureTarget.Texture2D)
        {
            m_handle = GL.GenTexture();
            Size = new Vector2I(width, height);
            m_target = target;
        }

        public virtual void Dispose()
        {
            GL.DeleteTexture(m_handle);
        }


        protected void Init(
            int[] data,
            PixelFormat dataFormat = PixelFormat.Bgra,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba,
            bool generateMipmap = false)
        {
            m_internalFormat = dataFormat;

            Bind();

            GL.TexImage2D(
                m_target,
                0,
                internalDataFormat,
                Size.X, Size.Y,
                0,
                dataFormat,
                PixelType.UnsignedByte,
                data);

            if (generateMipmap)
                GL.GenerateMipmap((GenerateMipmapTarget)m_target);
        }

        protected void InitMultisample(
            int multiSampleCount = 4,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba)
        {
            m_internalFormat = PixelFormat.Bgra; // not valid for multisample textures

            Bind();

            GL.TexImage2DMultisample(
                TextureTargetMultisample.Texture2DMultisample,
                multiSampleCount,
                internalDataFormat,
                Size.X, Size.Y,
                true);
        }

        protected void SetParameters(
            TextureMinFilter minFilter = TextureMinFilter.Linear,
            TextureMagFilter magFilter = TextureMagFilter.Linear,
            TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat)
        {
            Bind();

            GL.TexParameter(m_target, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(m_target, TextureParameterName.TextureMagFilter, (int)magFilter);
            GL.TexParameter(m_target, TextureParameterName.TextureWrapS, (int)wrapMode);
            GL.TexParameter(m_target, TextureParameterName.TextureWrapT, (int)wrapMode);
        }


        public void Update(Vector2I count, Vector2I offset = default(Vector2I), PixelType targetType = PixelType.UnsignedInt)
        {
            Update(count, offset, targetType, default(uint[]));
        }

        public void Update<T>(Vector2I count, Vector2I offset = default(Vector2I), PixelType targetType = PixelType.UnsignedInt, T[] data = null)
            where T : struct
        {
            Bind();
            GL.TexSubImage2D(m_target, 0, offset.X, offset.Y, count.X, count.Y, m_internalFormat, targetType, data);
        }


        public void Copy2D(PixelType targetType = PixelType.UnsignedInt)
        {
            Copy2D(targetType, default(uint[]));
        }

        public void Copy2D<T>(PixelType targetType = PixelType.UnsignedInt, T[] targetBuffer = null)
            where T : struct
        {
            Bind();
            GL.GetTexImage(m_target, 0, m_internalFormat, targetType, targetBuffer);
        }


        public virtual void Bind()
        {
            GL.BindTexture(m_target, m_handle);
        }
    }
}
