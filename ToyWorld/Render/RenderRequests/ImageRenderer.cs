using System;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using RenderingBase.Renderer;
using RenderingBase.RenderObjects.Buffers;
using TmxMapSerializer.Elements;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    public class ImageRenderer
        : IDisposable
    {
        #region Fields

        protected Pbo m_pbo;

        private ImageSettings m_settings;

        #endregion

        #region Genesis

        public virtual void Dispose()
        {
            if (m_pbo != null)
                m_pbo.Dispose();
        }

        #endregion

        #region Init

        public virtual void Init(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world, ImageSettings settings)
        {
            if (settings == null)
                return;

            switch (settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.OpenglPbo:
                    // Set up pixel buffer object for data transfer to RR issuer; don't allocate any memory (it's done in CheckDirtyParams)
                    m_pbo = new Pbo();
                    m_pbo.Init(Resolution.Width * Resolution.Height, null, BufferUsageHint.StreamDraw);
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    settings.RenderedScene = new uint[Resolution.Width * Resolution.Height];
                    break;
            }
        }

        #endregion

        #region Draw

        public virtual void Draw(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            if (renderRequest.CopyToWindow)
            {
                // TODO: TEMP: copy to default framebuffer (our window) -- will be removed
                m_frontFbo.Bind(FramebufferTarget.ReadFramebuffer);
                GL.BindFramebuffer(FramebufferTarget.DrawFramebuffer, 0);
                GL.BlitFramebuffer(
                    0, 0, m_frontFbo.Size.X, m_frontFbo.Size.Y,
                    0, 0, renderer.Width, renderer.Height,
                    ClearBufferMask.ColorBufferBit,
                    BlitFramebufferFilter.Nearest);
            }

            // Gather data to host mem
            m_frontFbo.Bind();
            GL.ReadBuffer(ReadBufferMode.ColorAttachment0); // Works for fbo bound to Framebuffer (not DrawFramebuffer)

            switch (m_settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.OpenglPbo:
                    m_pbo.Bind();
                    GL.ReadPixels(0, 0, Resolution.Width, Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, default(IntPtr));
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                    GL.ReadPixels(0, 0, Resolution.Width, Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, m_settings.RenderedScene);
                    break;
            }
        }

        #endregion
    }
}
