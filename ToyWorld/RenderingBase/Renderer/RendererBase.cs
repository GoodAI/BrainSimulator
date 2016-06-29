using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using GoodAI.ToyWorld.Control;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Effects;
using RenderingBase.RenderObjects.Geometries;
using RenderingBase.RenderObjects.Textures;
using RenderingBase.RenderRequests;

namespace RenderingBase.Renderer
{
    public abstract class RendererBase<TWorld>
        : IDisposable
        where TWorld : class
    {
        #region RR comparer

        class RenderRequestComparer
            : IComparer<IRenderRequestBaseInternal<TWorld>>
        {
            public int Compare(IRenderRequestBaseInternal<TWorld> first, IRenderRequestBaseInternal<TWorld> second)
            {
                int resolutionFirst = first.Resolution.Width * first.Resolution.Height;
                int resolutionSecond = second.Resolution.Width * second.Resolution.Height;

                return Math.Sign(resolutionFirst - resolutionSecond);
            }
        }

        #endregion

        #region Fields

        public uint SimTime { get; private set; }

        private readonly List<IRenderRequestBaseInternal<TWorld>> m_renderRequestQueue = new List<IRenderRequestBaseInternal<TWorld>>();
        private readonly List<IRenderRequestBaseInternal<TWorld>> m_dirtyRenderRequestQueue = new List<IRenderRequestBaseInternal<TWorld>>();
        private RenderRequestComparer m_renderRequestComparer = new RenderRequestComparer();

        public readonly GeometryManager GeometryManager = new GeometryManager();
        public readonly EffectManager EffectManager = new EffectManager();
        public readonly TextureManager TextureManager = new TextureManager();
        public readonly RenderTargetManager RenderTargetManager = new RenderTargetManager();

        #endregion

        #region Genesis

        internal RendererBase()
        {
            StaticVboFactory.Init();
        }

        public virtual void Dispose()
        {
            // Dispose of RRs
            while (m_renderRequestQueue.Count > 0) // Cannot dispose in a foreach cycle (they remove themselves from the queue)
            {
                var rr = m_renderRequestQueue[m_renderRequestQueue.Count - 1];
                m_renderRequestQueue.RemoveAt(m_renderRequestQueue.Count - 1);
                rr.Dispose();

                Debug.Assert(!m_renderRequestQueue.Contains(rr));
            }

            StaticVboFactory.Clear();
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
            m_dirtyRenderRequestQueue.Clear();
        }

        public virtual void ProcessRequests()
        {
            // Init stuff
            SimTime++;
            MakeContextCurrent();


            // Init and add new RRs

            bool anyDirty = m_dirtyRenderRequestQueue.Count > 0;

            foreach (var dirtyRenderRequest in m_dirtyRenderRequestQueue)
            {
                dirtyRenderRequest.Init();
                m_renderRequestQueue.Add(dirtyRenderRequest);
            }

            m_dirtyRenderRequestQueue.Clear();

            if (anyDirty)
                m_renderRequestQueue.Sort(m_renderRequestComparer);


            // Process RRs
            foreach (IRenderRequestBaseInternal<TWorld> renderRequest in m_renderRequestQueue)
                renderRequest.Update();

            foreach (IRenderRequestBaseInternal<TWorld> renderRequest in m_renderRequestQueue)
                renderRequest.OnPreDraw();

            foreach (IRenderRequestBaseInternal<TWorld> renderRequest in m_renderRequestQueue)
            {
                Process(renderRequest);
                CheckError();
            }

            foreach (IRenderRequestBaseInternal<TWorld> renderRequest in m_renderRequestQueue)
                renderRequest.OnPostDraw();
        }

        protected virtual void Process(IRenderRequestBaseInternal<TWorld> request)
        {
            request.Draw();
        }

        [Conditional("DEBUG")]
        public virtual void CheckError()
        { }

        #endregion


        public void EnqueueRequest(IRenderRequest request)
        {
            m_dirtyRenderRequestQueue.Add((IRenderRequestBaseInternal<TWorld>)request);
        }

        public void EnqueueRequest(IAvatarRenderRequest request)
        {
            m_dirtyRenderRequestQueue.Add((IRenderRequestBaseInternal<TWorld>)request);
        }

        public void RemoveRenderRequest(IRenderRequestBaseInternal<TWorld> request)
        {
            m_renderRequestQueue.Remove(request);
            m_dirtyRenderRequestQueue.Remove(request);
        }
    }
}
