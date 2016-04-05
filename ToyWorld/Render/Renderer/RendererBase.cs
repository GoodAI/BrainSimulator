using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using Render.RenderRequests;
using Render.RenderRequests.AvatarRenderRequests;
using Render.RenderRequests.RenderRequests;
using System;
using System.Diagnostics;
using Render.Shaders;
using VRage.Collections;

namespace Render.Renderer
{
    public abstract class RendererBase : IRenderer
    {
        #region Fields

        private readonly IterableQueue<RenderRequest> m_renderRequestQueue = new IterableQueue<RenderRequest>();

        internal Shader CurrentShader;

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

        #region IRenderer overrides

        public INativeWindow Window { get; protected set; }
        public IGraphicsContext Context { get; protected set; }

        public virtual void CreateWindow(string title, int width, int height)
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

        public virtual void CreateContext()
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

        public virtual void Init()
        {
            m_renderRequestQueue.Clear();
        }

        public virtual void Reset()
        {
            m_renderRequestQueue.Clear();
        }

        public virtual void EnqueueRequest(IRenderRequest request)
        {
            m_renderRequestQueue.Enqueue((RenderRequestBase)request);
        }

        public virtual void EnqueueRequest(IAvatarRenderRequest request)
        {
            m_renderRequestQueue.Enqueue((AvatarRenderRequestBase)request);
        }

        public virtual void ProcessRequests()
        {
            Window.ProcessEvents();

            MakeContextCurrent();

            foreach (var renderRequest in m_renderRequestQueue)
                Process(renderRequest);

            CheckError();

            Context.MakeCurrent(null);
        }

        protected virtual void Process(RenderRequest request)
        {
            request.Draw(this);
        }

        #endregion


        public void MakeContextCurrent()
        {
            if (!Context.IsCurrent)
                Context.MakeCurrent(Window.WindowInfo);
        }

        public void MakeContextNotCurrent()
        {
            if (Context.IsCurrent)
                Context.MakeCurrent(null);
        }


        [Conditional("DEBUG")]
        public void CheckError()
        {
            int i = 60;
            ErrorCode err;

            try
            {
                while ((err = GL.GetError()) != ErrorCode.NoError)
                {
                    // TODO: log
                    if (--i == 0)
                        throw new Exception(err.ToString());
                }
            }
            catch (Exception e)
            {
                throw;
            }
        }
    }
}
