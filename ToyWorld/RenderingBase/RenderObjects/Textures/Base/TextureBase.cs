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


        #region Genesis

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

        #endregion

        #region Init

        public void Init1D(
            PixelFormat dataFormat = PixelFormat.Bgra,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba,
            PixelType dataType = PixelType.UnsignedByte,
            bool generateMipmap = false)
        {
            Init1D(default(int[]), dataFormat, internalDataFormat, dataType, generateMipmap);
        }

        public void Init1D<T>(
            T[] data,
            PixelFormat dataFormat = PixelFormat.Bgra,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba,
            PixelType dataType = PixelType.UnsignedByte,
            bool generateMipmap = false)
            where T : struct
        {
            m_internalFormat = dataFormat;

            Bind();

            GL.TexImage1D(m_target, 0, internalDataFormat, Size.Size(), 0, dataFormat, dataType, data);

            if (generateMipmap)
                GL.GenerateMipmap((GenerateMipmapTarget)m_target);
        }

        public void Init2D(
            PixelFormat dataFormat = PixelFormat.Bgra,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba,
            PixelType dataType = PixelType.UnsignedByte,
            bool generateMipmap = false)
        {
            Init2D(default(int[]), dataFormat, internalDataFormat, dataType, generateMipmap);
        }

        public void Init2D<T>(
            T[] data,
            PixelFormat dataFormat = PixelFormat.Bgra,
            PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba,
            PixelType dataType = PixelType.UnsignedByte,
            bool generateMipmap = false)
            where T : struct
        {
            m_internalFormat = dataFormat;

            Bind();

            GL.TexImage2D(m_target, 0, internalDataFormat, Size.X, Size.Y, 0, dataFormat, dataType, data);

            if (generateMipmap)
                GL.GenerateMipmap((GenerateMipmapTarget)m_target);
        }

        public void InitMultisample(int multiSampleCount = 4, PixelInternalFormat internalDataFormat = PixelInternalFormat.Rgba)
        {
            m_internalFormat = PixelFormat.Bgra; // not valid for multisample textures

            Bind();

            GL.TexImage2DMultisample(TextureTargetMultisample.Texture2DMultisample, multiSampleCount, internalDataFormat, Size.X, Size.Y, true);
        }

        public void SetParameters(TextureMinFilter minFilter = TextureMinFilter.Linear, TextureMagFilter magFilter = TextureMagFilter.Linear, TextureWrapMode wrapMode = TextureWrapMode.MirroredRepeat)
        {
            Bind();

            GL.TexParameter(m_target, TextureParameterName.TextureMinFilter, (int)minFilter);
            GL.TexParameter(m_target, TextureParameterName.TextureMagFilter, (int)magFilter);
            GL.TexParameter(m_target, TextureParameterName.TextureWrapS, (int)wrapMode);
            GL.TexParameter(m_target, TextureParameterName.TextureWrapT, (int)wrapMode);
        }

        #endregion

        #region Updating

        public void Update1D(int count, int offset = 0, PixelType dataType = PixelType.UnsignedInt)
        {
            Update1D(count, offset, dataType, default(uint[]));
        }

        public void Update1D<T>(int count, int offset = 0, PixelType dataType = PixelType.UnsignedInt, T[] data = null) where T : struct
        {
            Bind();
            GL.TexSubImage1D(m_target, 0, offset, count, m_internalFormat, dataType, data);
        }

        public void Update2D(Vector2I count, Vector2I offset = default(Vector2I), PixelType dataType = PixelType.UnsignedInt)
        {
            Update2D(count, offset, dataType, default(uint[]));
        }

        public void Update2D<T>(Vector2I count, Vector2I offset = default(Vector2I), PixelType dataType = PixelType.UnsignedInt, T[] data = null) where T : struct
        {
            Bind();
            GL.TexSubImage2D(m_target, 0, offset.X, offset.Y, count.X, count.Y, m_internalFormat, dataType, data);
        }


        public void Copy2D(PixelType targetType = PixelType.UnsignedInt)
        {
            Copy2D(targetType, default(uint[]));
        }

        public void Copy2D<T>(PixelType targetType = PixelType.UnsignedInt, T[] targetBuffer = null) where T : struct
        {
            Bind();
            GL.GetTexImage(m_target, 0, m_internalFormat, targetType, targetBuffer);
        }

        #endregion


        public virtual void Bind()
        {
            GL.BindTexture(m_target, m_handle);
        }
    }
}
