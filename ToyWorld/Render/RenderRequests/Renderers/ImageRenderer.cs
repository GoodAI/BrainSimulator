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

        protected Pbo Pbo;
        protected uint[] RenderedScene;

        #endregion

        #region Genesis

        public ImageRenderer(RenderRequest owner)
            : base(owner)
        { }

        public virtual void Dispose()
        {
            if (Pbo != null)
                Pbo.Dispose();
        }

        #endregion

        #region Init

        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world, ImageSettings settings)
        {
            Settings = settings;
            int bufferSize = Owner.Resolution.Width * Owner.Resolution.Height;

            switch (Settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.OpenglPbo:
                    // Set up pixel buffer object for data transfer to RR issuer; don't allocate any memory (it's done in CheckDirtyParams)
                    Pbo = new Pbo();
                    Pbo.Init(bufferSize, null, BufferUsageHint.StreamDraw);
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    if (RenderedScene == null || RenderedScene.Length < bufferSize)
                        RenderedScene = new uint[bufferSize];
                    break;
            }
        }

        #endregion

        #region Callbacks

        public virtual void OnPreDraw()
        {
            if (Settings.CopyMode == RenderRequestImageCopyingMode.OpenglPbo)
                Settings.InvokePreRenderingEvent(Owner, Pbo.Handle);
        }

        public virtual void OnPostDraw()
        {
            if (Settings.CopyMode == RenderRequestImageCopyingMode.OpenglPbo)
                Settings.InvokePostRenderingEvent(Owner, Pbo.Handle);
        }

        #endregion

        #region Draw

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            if (Settings.CopyMode == RenderRequestImageCopyingMode.None)
                return;

            // Gather data to host mem
            Owner.FrontFbo.Bind();
            GL.ReadBuffer(ReadBufferMode.ColorAttachment0); // Works for fbo bound to Framebuffer (not DrawFramebuffer)

            switch (Settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.DefaultFbo:
                    //Owner.FrontFbo.Bind(FramebufferTarget.ReadFramebuffer);
                    GL.BindFramebuffer(FramebufferTarget.DrawFramebuffer, 0);
                    GL.BlitFramebuffer(
                        0, 0, Owner.FrontFbo.Size.X, Owner.FrontFbo.Size.Y,
                        0, 0, renderer.Width, renderer.Height,
                        ClearBufferMask.ColorBufferBit,
                        BlitFramebufferFilter.Nearest);
                    break;
                case RenderRequestImageCopyingMode.OpenglPbo:
                    Pbo.Bind();
                    GL.ReadPixels(0, 0, Owner.Resolution.Width, Owner.Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, default(IntPtr));
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                    GL.ReadPixels(0, 0, Owner.Resolution.Width, Owner.Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, RenderedScene);
                    Settings.InvokePostBufferPrepared(Owner, RenderedScene);
                    break;
            }
        }

        #endregion
    }
}
