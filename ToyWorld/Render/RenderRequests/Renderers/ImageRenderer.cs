using System;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using RenderingBase.Renderer;
using RenderingBase.RenderObjects.Buffers;
using TmxMapSerializer.Elements;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class ImageRenderer
        : RRRendererBase<ImageSettings, RenderRequest>, IDisposable
    {
        #region Fields

        protected Pbo m_pbo;

        #endregion

        #region Genesis

        public virtual void Dispose()
        {
            if (m_pbo != null)
                m_pbo.Dispose();
        }

        #endregion

        #region Init

        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world, RenderRequest renderRequest, ImageSettings settings)
        {
            if (settings == null)
                return;

            switch (settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.OpenglPbo:
                    // Set up pixel buffer object for data transfer to RR issuer; don't allocate any memory (it's done in CheckDirtyParams)
                    m_pbo = new Pbo();
                    m_pbo.Init(Owner.Resolution.Width * Owner.Resolution.Height, null, BufferUsageHint.StreamDraw);
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    settings.RenderedScene = new uint[Owner.Resolution.Width * Owner.Resolution.Height];
                    break;
            }
        }

        #endregion

        #region Callbacks

        public virtual void OnPreDraw()
        {
            if (Settings.CopyMode == RenderRequestImageCopyingMode.OpenglPbo)
                Settings.InvokePreRenderingEvent(Owner, m_pbo.Handle);
        }

        public virtual void OnPostDraw()
        {
            if (Settings.CopyMode == RenderRequestImageCopyingMode.OpenglPbo)
                Settings.InvokePostRenderingEvent(Owner, m_pbo.Handle);
        }

        #endregion

        #region Draw

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            if (Owner.CopyToWindow)
            {
                // TODO: TEMP: copy to default framebuffer (our window) -- will be removed
                Owner.FrontFbo.Bind(FramebufferTarget.ReadFramebuffer);
                GL.BindFramebuffer(FramebufferTarget.DrawFramebuffer, 0);
                GL.BlitFramebuffer(
                    0, 0, Owner.FrontFbo.Size.X, Owner.FrontFbo.Size.Y,
                    0, 0, renderer.Width, renderer.Height,
                    ClearBufferMask.ColorBufferBit,
                    BlitFramebufferFilter.Nearest);
            }

            // Gather data to host mem
            Owner.FrontFbo.Bind();
            GL.ReadBuffer(ReadBufferMode.ColorAttachment0); // Works for fbo bound to Framebuffer (not DrawFramebuffer)

            switch (Settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.OpenglPbo:
                    m_pbo.Bind();
                    GL.ReadPixels(0, 0, Owner.Resolution.Width, Owner.Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, default(IntPtr));
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                    GL.ReadPixels(0, 0, Owner.Resolution.Width, Owner.Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, Settings.RenderedScene);
                    break;
            }
        }

        #endregion
    }
}
