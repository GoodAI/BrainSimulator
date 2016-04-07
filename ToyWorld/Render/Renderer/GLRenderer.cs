using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using Render.RenderRequests;
using Render.RenderRequests.AvatarRenderRequests;
using Render.RenderRequests.RenderRequests;
using System;
using System.Diagnostics;
using Render.RenderObjects.Effects;
using Render.RenderObjects.Geometries;
using Render.RenderObjects.Textures;
using VRage.Collections;

namespace Render.Renderer
{
    public class GLRenderer : RendererBase
    {
        #region Genesis

        public GLRenderer()
        { }

        public override void Dispose()
        {
            base.Dispose();

            // Dispose of Context
            if (Context != null)
            {
                if (Context.IsCurrent)
                    Context.MakeCurrent(null);
                Context.Dispose();
                Context = null;
            }

            // Dispose of Window
            if (Window != null)
            {
                Window.Close();
                Window.Dispose();
                Window = null;
            }
        }

        #endregion

        #region RendererBase overrides

        public override int Width { get { return Window.Width; } }
        public override int Height { get { return Window.Height; } }

        public INativeWindow Window { get; set; }
        public IGraphicsContext Context { get; set; }

        public override void CreateWindow(string title, int width, int height)
        {
            if (Window != null)
            {
                Window.Close();
                Window.Dispose();
            }

            Window = new NativeWindow(width, height, title, GameWindowFlags.Default, GraphicsMode.Default, DisplayDevice.Default);
            Window.Resize += WindowOnResize;
        }

        private void WindowOnResize(object sender, EventArgs args)
        {
            if (Context != null)
                Context.Update(Window.WindowInfo);
        }

        public override void CreateContext()
        {
            Debug.Assert(Window != null, "Missing window, cannot create context.");

            if (Context != null)
            {
                MakeContextNotCurrent();
                Context.Dispose();
            }

            Context = new GraphicsContext(GraphicsMode.Default, Window.WindowInfo);
            Context.LoadAll();
            Context.MakeCurrent(null);
        }

        public override void MakeContextCurrent()
        {
            if (!Context.IsCurrent)
                Context.MakeCurrent(Window.WindowInfo);
        }

        public override void MakeContextNotCurrent()
        {
            if (Context.IsCurrent)
                Context.MakeCurrent(null);
        }

        public override void ProcessRequests()
        {
            Window.ProcessEvents();

            base.ProcessRequests();

            CheckError();
        }

        public override void CheckError()
        {
            int i = 60;
            ErrorCode err;

            while ((err = GL.GetError()) != ErrorCode.NoError)
            {
                // TODO: log
                if (--i == 0)
                    throw new Exception(err.ToString());
            }
        }

        #endregion

    }
}
