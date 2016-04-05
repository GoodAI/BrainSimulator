using GoodAI.ToyWorld.Control;
using Render.RenderRequests;
using Render.RenderRequests.AvatarRenderRequests;
using Render.RenderRequests.RenderRequests;
using System;
using System.Diagnostics;
using Render.RenderObjects.Effects;
using Render.RenderObjects.Geometries;
using Render.RenderObjects.Shaders;
using Render.RenderObjects.Textures;
using VRage.Collections;

namespace Render.Renderer
{
    public abstract class RendererBase : IDisposable
    {
        #region Fields

        private readonly IterableQueue<RenderRequest> m_renderRequestQueue = new IterableQueue<RenderRequest>();

        internal readonly GeometryManager GeometryManager = new GeometryManager();
        internal readonly EffectManager EffectManager = new EffectManager();
        internal readonly TextureManager TextureManager = new TextureManager();

        #endregion

        #region Genesis

        internal RendererBase()
        { }

        public virtual void Dispose()
        {
            // Dispose of RRs
            foreach (var renderRequest in m_renderRequestQueue)
                renderRequest.Dispose();

            m_renderRequestQueue.Clear();
        }

        #endregion

        #region Virtual stuff

        public abstract int Width { get; }
        public abstract int Height { get; }

        public abstract void CreateWindow(string title, int width, int height);
        public abstract void CreateContext();
        public abstract void MakeContextCurrent();
        public abstract void MakeContextNotCurrent();

        public virtual void Init()
        {
            m_renderRequestQueue.Clear();
        }

        public virtual void Reset()
        {
            m_renderRequestQueue.Clear();
        }

        public virtual void ProcessRequests()
        {
            MakeContextCurrent();

            foreach (var renderRequest in m_renderRequestQueue)
                Process(renderRequest);
        }

        protected virtual void Process(RenderRequest request)
        {
            request.Draw(this);
        }

        #endregion


        public void EnqueueRequest(IRenderRequest request)
        {
            m_renderRequestQueue.Enqueue((RenderRequestBase)request);
        }

        public void EnqueueRequest(IAvatarRenderRequest request)
        {
            m_renderRequestQueue.Enqueue((AvatarRenderRequestBase)request);
        }
    }
}
