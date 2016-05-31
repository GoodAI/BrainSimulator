using GoodAI.ToyWorld.Control;
using Render.RenderRequests;
using RenderingBase.Renderer;
using RenderingBase.RenderRequests;
using World.ToyWorldCore;

namespace Render.Renderer
{
    public class ToyWorldRenderer
        : GLRenderer<ToyWorld>
    {
        static ToyWorldRenderer()
        {
            //////////////////////
            // NOTE: All renderRequests must inherit from RenderRequest
            //////////////////////

            RenderRequestFactory.CaseInternal<IFullMapRR, FullMapRR>();
            RenderRequestFactory.CaseInternal<IFreeMapRR, FreeMapRR>();

            RenderRequestFactory.CaseParamInternal<IFovAvatarRR, FovAvatarRR>();
            RenderRequestFactory.CaseParamInternal<IFofAvatarRR, FofAvatarRR>();
        }
    }
}
