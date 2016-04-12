using System;
using GoodAI.ToyWorld.Control;
using Render.Renderer;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    public abstract class RenderRequest : IDisposable
    {
        public virtual void Dispose()
        { }


        public abstract void Init(RendererBase renderer, ToyWorld world);
        public abstract void Draw(RendererBase renderer, ToyWorld world);
    }
}
