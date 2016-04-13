using System;
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
            // TODO: gather and distribute types to TypeSwitches based on available constructor through reflection (add attributes?)
            // Plain geometries
            CaseInternal<FullScreenQuad>();
            CaseInternal<FancyFullscreenQuad>();
            CaseInternal<FullScreenQuadTex>();
            CaseInternal<FullScreenQuadOffset>();

            // Parameterized geometries
            CaseParamInternal<FullScreenGrid>();
        }

        private GeometryManager CaseInternal<T>()
            where T : GeometryBase, new()
        {
            m_geometries.Case<T>(() => new T());
            return this;
        }

        private GeometryManager CaseParamInternal<T>()
            where T : GeometryBase
        {
            // Activator is about 11 times slower, than new T() -- should be ok for this usage
            m_vecGeometries.Case<T>(vec => (T)Activator.CreateInstance(typeof(T), vec));
            return this;
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
