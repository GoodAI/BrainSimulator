using Render.Tests.Geometries;
using Utils.VRageRIP.Lib.Collections;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    internal class GeometryManager
    {
        private readonly TypeSwitch<GeometryBase> m_geometries = new TypeSwitch<GeometryBase>();
        private readonly TypeSwitchParam<GeometryBase, Vector2I> m_vecGeometries = new TypeSwitchParam<GeometryBase, Vector2I>();


        public GeometryManager()
        {
            m_geometries
                .Case<FullScreenQuad>(() =>
                    new FullScreenQuad())
                .Case<FancyFullscreenQuad>(() =>
                    new FancyFullscreenQuad())
                .Case<FullScreenQuadTex>(() =>
                    new FullScreenQuadTex());

            m_vecGeometries
                .Case<FullScreenGrid>(vec =>
                    new FullScreenGrid(vec));
        }


        public T Get<T>()
            where T : GeometryBase
        {
            return m_geometries.Switch<T>();
        }

        public T Get<T>(Vector2I param)
            where T : GeometryBase
        {
            return m_vecGeometries.Switch<T>(param);
        }
    }
}
