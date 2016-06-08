using RenderingBase.Renderer;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal abstract class RRRendererBase<TSettings, TOwner>
        where TSettings : class
        where TOwner : class
    {
        protected TSettings Settings;
        protected TOwner Owner;


        public abstract void Init(RendererBase<ToyWorld> renderer, ToyWorld world, TOwner owner, TSettings settings);
        public abstract void Draw(RendererBase<ToyWorld> renderer, ToyWorld world);
    }
}
