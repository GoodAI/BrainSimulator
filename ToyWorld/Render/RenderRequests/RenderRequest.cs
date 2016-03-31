using System;
using GoodAI.ToyWorld.Control;
using Render.Renderer;

namespace Render.RenderRequests
{
    public abstract class RenderRequest : IDisposable
    {
        public virtual void Dispose()
        { }


        public abstract void Init(IRenderer renderer);
        public abstract void Draw(RendererBase renderer);
    }
}
