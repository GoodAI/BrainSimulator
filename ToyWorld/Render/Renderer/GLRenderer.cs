using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using World.ToyWorldCore;

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

        public override void ProcessRequests(ToyWorld world)
        {
            Window.ProcessEvents();

            base.ProcessRequests(world);
        }

        public override void CheckError()
        {
            base.CheckError();

            ErrorCode error, previousError;

            if ((previousError = GL.GetError()) != ErrorCode.NoError)
                Debug.Fail("GL error: " + previousError);

            while ((error = GL.GetError()) != ErrorCode.NoError)
            {
                // TODO: log

                if (previousError == error)
                {
                    // the same error repeated twice indicates an infinite loop of errors
                    throw new Exception(error.ToString());
                }

                previousError = error;
            }
        }

        #endregion
    }
}
