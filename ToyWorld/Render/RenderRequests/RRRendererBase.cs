using System.Diagnostics;
using RenderingBase.Renderer;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal abstract class RRRendererBase<TSettings, TOwner>
        where TSettings : class
        where TOwner : class
    {
        internal TSettings Settings;
        protected readonly TOwner Owner;


        protected RRRendererBase(TOwner owner)
        {
            Debug.Assert(owner != null);
            Owner = owner;
        }


        public abstract void Init(RendererBase<ToyWorld> renderer, ToyWorld world, TSettings settings);
        public abstract void Draw(RendererBase<ToyWorld> renderer, ToyWorld world);
    }
}
