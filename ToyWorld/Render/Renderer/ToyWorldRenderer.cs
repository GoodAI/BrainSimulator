using GoodAI.ToyWorld.Control;
using Render.RenderObjects.Effects;
using Render.RenderObjects.Geometries;
using Render.RenderRequests;
using RenderingBase.Renderer;
using RenderingBase.RenderObjects.Geometries;
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

            //// RenderRequest registration
            RenderRequestFactory.CaseInternal<IFullMapRR, FullMapRR>();
            RenderRequestFactory.CaseInternal<IFreeMapRR, FreeMapRR>();

            RenderRequestFactory.CaseParamInternal<IFovAvatarRR, FovAvatarRR>();
            RenderRequestFactory.CaseParamInternal<IFofAvatarRR, FofAvatarRR>();
        }

        public ToyWorldRenderer()
        {
            // TODO: gather and distribute types to TypeSwitches based on available constructor through reflection (add attributes?)

            //// Geometry registration
            // Plain geometries
            GeometryManager.Case<FullScreenQuad>();
            GeometryManager.Case<FullScreenQuadTex>();
            GeometryManager.Case<FullScreenQuadOffset>();

            // Parameterized geometries
            GeometryManager.CaseParam<FullScreenGrid>();
            GeometryManager.CaseParam<FullScreenGridTex>();


            EffectManager.Case<NoEffectTex>();
            EffectManager.Case<NoEffectOffset>();
            EffectManager.Case<SmokeEffect>();
            EffectManager.Case<NoiseEffect>();
        }
    }
}
