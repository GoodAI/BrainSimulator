using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RenderingBase.Renderer;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal abstract class RRRendererBase<TSettings>
        where TSettings : class
    {
        protected TSettings Settings;


        public abstract void Init(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world, TSettings settings);
        public abstract void Draw(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world);
    }
}
