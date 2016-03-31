using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using Render.RenderRequests;
using Render.RenderRequests.AvatarRenderRequests;
using Render.RenderRequests.RenderRequests;
using VRage.Collections;

namespace Render.Renderer
{
    public abstract class RendererBase : IRenderer
    {
        #region Fields

        private readonly IterableQueue<RenderRequest> m_renderRequestQueue = new IterableQueue<RenderRequest>();

        #endregion

        #region Genesis
        internal RendererBase()
        { }

        public void Dispose()
        {
            // Dispose of RRs
            foreach (var renderRequest in m_renderRequestQueue)
                renderRequest.Dispose();

            m_renderRequestQueue.Clear();

            // Dispose of Context
            if (Context.IsCurrent)
                Context.MakeCurrent(null);
            Context.Dispose();
            Context = null;

            // Dispose of Window
            Window.Close();
            Window.Dispose();
            Window = null;
        }

        #endregion

        #region IRenderer overrides

        public INativeWindow Window { get; protected set; }
        public IGraphicsContext Context { get; protected set; }

        public virtual void CreateWindow(string title, int width, int height)
        {
            Window = new NativeWindow(width, height, title, GameWindowFlags.Default, GraphicsMode.Default, DisplayDevice.Default);
            Window.Resize += WindowOnResize;
        }

        private void WindowOnResize(object sender, EventArgs args)
        {
            if (Context != null)
                Context.Update(Window.WindowInfo);
        }

        public virtual void CreateContext()
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

        public virtual void Init()
        {
            m_renderRequestQueue.Clear();
        }

        public virtual void Reset()
        {
            Init();
        }

        public virtual void EnqueueRequest(IRenderRequest request)
        {
            Debug.Assert(request != null);
            Debug.Assert(request is RenderRequestBase);
            m_renderRequestQueue.Enqueue((RenderRequestBase)request);
            CheckError();
        }

        public virtual void EnqueueRequest(IAvatarRenderRequest request)
        {
            Debug.Assert(request != null);
            Debug.Assert(request is AvatarRenderRequestBase);
            m_renderRequestQueue.Enqueue((AvatarRenderRequestBase)request);
            CheckError();
        }

        public virtual void ProcessRequests()
        {
            Debug.Assert(Context != null);

            Window.ProcessEvents();

            if (!Context.IsCurrent)
                Context.MakeCurrent(Window.WindowInfo);

            foreach (var renderRequest in m_renderRequestQueue)
                Process(renderRequest);

            Context.MakeCurrent(null);
        }

        protected virtual void Process(RenderRequest request)
        {
            request.Draw(this);
        }

        #endregion


        [Conditional("DEBUG")]
        void CheckError()
        {
            var err = GL.GetError();
            Debug.Assert(err == ErrorCode.NoError, err.ToString());
            // TODO: log
        }
    }
}
