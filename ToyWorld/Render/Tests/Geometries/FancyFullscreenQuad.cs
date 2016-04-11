using Render.RenderObjects.Buffers;
using Render.RenderObjects.Geometries;

namespace Render.Tests.Geometries
{
    internal class FancyFullscreenQuad : FullScreenQuad
    {
        public FancyFullscreenQuad()
        {
            this[VboPosition.Colors] = StaticVboFactory.QuadColors;
            EnableAttrib(VboPosition.Colors);
        }
    }
}
