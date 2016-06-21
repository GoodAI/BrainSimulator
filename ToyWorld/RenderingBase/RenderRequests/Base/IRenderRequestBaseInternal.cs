using System;
using GoodAI.ToyWorld.Control;
using RenderingBase.Renderer;

namespace RenderingBase.RenderRequests
{
    public interface IRenderRequestBaseInternal<TWorld>
        : IRenderRequestBase
        where TWorld : class
    {
        RendererBase<TWorld> Renderer { get; set; }
        TWorld World { get; set; }

        void Init();
        void Draw();

        void OnPreDraw();
        void OnPostDraw();
    }
}
