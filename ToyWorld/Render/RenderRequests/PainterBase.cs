using System.Diagnostics;
using RenderingBase.Renderer;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal abstract class PainterBase<TSettings, TOwner>
        where TSettings : struct 
        where TOwner : class
    {
        internal TSettings Settings;
        protected readonly TOwner Owner;


        protected PainterBase(TOwner owner)
        {
            Debug.Assert(owner != null);
            Owner = owner;
        }


        public abstract void Init(RendererBase<ToyWorld> renderer, ToyWorld world, TSettings settings);
        public abstract void Draw(RendererBase<ToyWorld> renderer, ToyWorld world);
    }
}
