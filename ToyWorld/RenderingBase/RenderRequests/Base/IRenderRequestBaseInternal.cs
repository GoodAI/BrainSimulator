using System;
using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;

namespace RenderingBase.RenderRequests
{
    public interface IRenderRequestBaseInternal<TWorld>
        : IRenderRequestBase, IDisposable
        where TWorld : class
    {
        bool CopyToWindow { get; set; }

        void Init(RendererBase<TWorld> renderer, TWorld world);
        void Draw(RendererBase<TWorld> renderer, TWorld world);

        void OnPreDraw();
        void OnPostDraw();
    }
}
