using GoodAI.ToyWorld.Control;
using RenderingBase.RenderRequests;

namespace Render.RenderRequests
{
    public abstract class RRBase : RenderRequest, IRenderRequest
    {
        static RRBase()
        {
            //////////////////////
            // NOTE: All renderRequests must inherit from RenderRequest
            //////////////////////

            RenderRequestFactory.CaseParamInternal<IFovAvatarRR, FovAvatarRR>();
            RenderRequestFactory.CaseParamInternal<IFofAvatarRR, FofAvatarRR>();
        }
    }
}
