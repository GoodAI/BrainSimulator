using Render.Tests.Geometries;
using Utils.VRageRIP.Lib.Collections;

namespace Render.RenderObjects.Geometries
{
    internal class GeometryManager
    {
        private readonly TypeSwitch<GeometryBase> m_effects = new TypeSwitch<GeometryBase>();


        public GeometryManager()
        {
            m_effects
                .Case<FullScreenQuad>(() =>
                    new FullScreenQuad())
                .Case<FancyFullscreenQuad>(() =>
                    new FancyFullscreenQuad());
        }


        public void Draw<T>()
            where T : GeometryBase
        {
            m_effects.Switch<T>().Draw();
        }

        // update
    }
}
