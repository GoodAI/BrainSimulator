using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using GoodAI.ToyWorld.Render;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace Render.Renderer
{
    public class GLRenderer : IRenderer
    {
        #region Fields

        private readonly Queue<IRenderRequest> m_renderRequestQueue = new Queue<IRenderRequest>();

        #endregion

        #region IRenderer overrides

        public INativeWindow Window { get; protected set; }
        public IGraphicsContext Context { get; protected set; }

        public void CreateWindow(int width, int height)
        {
            Window = new NativeWindow(width, height, Utils.Globals.AppName, GameWindowFlags.Default, GraphicsMode.Default, DisplayDevice.Default);
            Window.Resize += WindowOnResize;
        }

        private void WindowOnResize(object sender, EventArgs args)
        {
            if (Context != null)
                Context.Update(Window.WindowInfo);
        }

        public void CreateContext()
        {
            Debug.Assert(Window != null, "Missing window, cannot create context.");

            if (Context != null)
            {
                Context.MakeCurrent(null);
                Context.Dispose();
            }

            Context = new GraphicsContext(GraphicsMode.Default, Window.WindowInfo);
            // Context.LoadAll(); // Is only needed when using an external context
        }

        public void Init()
        {
            m_renderRequestQueue.Clear();

            GL.ClearColor(Color.Black);

        }

        public void Reset()
        {

            Init();
        }

        public void EnqueueRequest(IRenderRequest request)
        {
            Debug.Assert(request != null);
            m_renderRequestQueue.Enqueue(request);
        }

        public void ProcessRequests()
        {
            Context.MakeCurrent(Window.WindowInfo);

            foreach (var renderRequest in m_renderRequestQueue)
            {
                Draw(renderRequest);
            }

            Context.MakeCurrent(null);
        }

        #endregion

        #region IDisposable overrides

        public void Dispose()
        {
            Context.MakeCurrent(null);
            Context.Dispose();
            Context = null;

            Window.Dispose();
            Window = null;
        }

        #endregion


        void Draw(IRenderRequest request)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            // Context.SwapBuffers();
        }

    }
}
