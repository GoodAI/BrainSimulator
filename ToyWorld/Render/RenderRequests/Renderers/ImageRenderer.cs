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

        protected Pbo<uint> Pbo;
        protected Pbo<float> DepthPbo;
        protected uint[] RenderedScene;
        protected float[] RenderedSceneDepth;

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
                    // Set up pixel buffer object for data transfer to RR issuer
                    Pbo = new Pbo<uint>();
                    Pbo.Init(bufferSize, null, BufferUsageHint.StreamDraw);

                    if (Settings.CopyDepth)
                    {
                        DepthPbo = new Pbo<float>();
                        DepthPbo.Init(bufferSize, null, BufferUsageHint.StreamDraw);
                    }
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    if (RenderedScene == null || RenderedScene.Length < bufferSize)
                    {
                        RenderedScene = new uint[bufferSize];

                        if (Settings.CopyDepth)
                            RenderedSceneDepth = new float[bufferSize];
                    }
                    break;
            }
        }

        #endregion

        #region Callbacks

        public virtual void OnPreDraw()
        {
            if (Settings.CopyMode == RenderRequestImageCopyingMode.OpenglPbo)
                Settings.InvokePreRenderingEvent(Owner, Pbo.Handle, Settings.CopyDepth ? DepthPbo.Handle : 0);
        }

        public virtual void OnPostDraw()
        {
            switch (Settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.OpenglPbo:
                    Settings.InvokePostRenderingEvent(Owner, Pbo.Handle, Settings.CopyDepth ? DepthPbo.Handle : 0);
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    Settings.InvokePostBufferPrepared(Owner, RenderedScene, Settings.CopyDepth ? RenderedSceneDepth : null);
                    break;
            }
        }

        #endregion

        #region Draw

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            if (Settings.CopyMode == RenderRequestImageCopyingMode.None)
                return;

            // Gather data to host mem
            Owner.FrontFbo.Bind();

            switch (Settings.CopyMode)
            {
                case RenderRequestImageCopyingMode.DefaultFbo:
                    GL.BindFramebuffer(FramebufferTarget.DrawFramebuffer, 0);
                    GL.BlitFramebuffer(
                        0, 0, Owner.FrontFbo.Size.X, Owner.FrontFbo.Size.Y,
                        0, 0, renderer.Width, renderer.Height,
                        ClearBufferMask.ColorBufferBit,
                        BlitFramebufferFilter.Nearest);
                    break;
                case RenderRequestImageCopyingMode.OpenglPbo:
                    Pbo.Bind();
                    GL.ReadBuffer(ReadBufferMode.ColorAttachment0); // Works for fbo bound to Framebuffer (not DrawFramebuffer)
                    //Owner.FrontFbo[FramebufferAttachment.ColorAttachment0].Copy2D(PixelType.UnsignedByte); // This is half as fast as ReadPixels for color data
                    GL.ReadPixels(0, 0, Owner.Resolution.Width, Owner.Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, default(IntPtr));

                    if (Settings.CopyDepth)
                    {
                        DepthPbo.Bind();
                        Owner.FrontFbo[FramebufferAttachment.DepthAttachment].Copy2D(); // This is twice as fast as ReadPixels for depth texture
                        //GL.ReadPixels(0, 0, Owner.Resolution.Width, Owner.Resolution.Height, PixelFormat.DepthComponent, PixelType.UnsignedInt, default(IntPtr));
                    }
                    break;
                case RenderRequestImageCopyingMode.Cpu:
                    GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                    GL.ReadBuffer(ReadBufferMode.ColorAttachment0); // Works for fbo bound to Framebuffer (not DrawFramebuffer)
                    GL.ReadPixels(0, 0, Owner.Resolution.Width, Owner.Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, RenderedScene);

                    if (Settings.CopyDepth)
                        Owner.FrontFbo[FramebufferAttachment.DepthAttachment].Copy2D(PixelType.UnsignedInt, RenderedSceneDepth);
                    break;
            }
        }

        #endregion
    }
}
