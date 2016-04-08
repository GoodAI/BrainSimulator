using Render.Tests.Geometries;
using Utils.VRageRIP.Lib.Collections;

namespace Render.RenderObjects.Geometries
{
    internal class GeometryManager
    {
        private readonly TypeSwitch<GeometryBase> m_geometries = new TypeSwitch<GeometryBase>();


        public GeometryManager()
        {
            m_geometries
                .Case<FullScreenQuad>(() =>
                    new FullScreenQuad())
                .Case<FancyFullscreenQuad>(() =>
                    new FancyFullscreenQuad())
                .Case<FullScreenQuadTex>(() =>
                    new FullScreenQuadTex());
        }


        public T Get<T>()
            where T : GeometryBase
        {
            return m_geometries.Switch<T>();
        }
    }
}
