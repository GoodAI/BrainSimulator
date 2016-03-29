using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using Render.RenderRequests;
using VRage.Collections;

namespace Render.Renderer
{
    public class GLRenderer : IRenderer
    {
        #region Fields

        private readonly IterableQueue<RenderRequestBase> m_renderRequestQueue = new IterableQueue<RenderRequestBase>();

        #endregion

        #region IRenderer overrides

        public INativeWindow Window { get; protected set; }
        public IGraphicsContext Context { get; protected set; }

        public void CreateWindow(string title, int width, int height)
        {
            Window = new NativeWindow(width, height, title, GameWindowFlags.Default, GraphicsMode.Default, DisplayDevice.Default);
            Window.Resize += WindowOnResize;
            Window.Visible = true;
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
            Context.LoadAll();
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
            Debug.Assert(request is RenderRequestBase);
            m_renderRequestQueue.Enqueue((RenderRequestBase)request);
        }

        public void EnqueueRequest(IAgentRenderRequest request)
        {
            // TODO
        }

        public void ProcessRequests()
        {
            Debug.Assert(Context != null);

            Context.MakeCurrent(Window.WindowInfo);

            foreach (var renderRequest in m_renderRequestQueue)
                Process(renderRequest);

            Context.MakeCurrent(null);
        }

        void Process(RenderRequestBase request)
        {
            request.Draw(this);
        }

        #endregion

        #region IDisposable overrides

        public void Dispose()
        {
            Context.MakeCurrent(null);
            Context.Dispose();
            Context = null;

            Window.Close();
            Window.Dispose();
            Window = null;
        }

        #endregion
    }
}
