using Render.RenderObjects.Buffers;
using Render.RenderObjects.Geometries;

namespace Render.Tests.Geometries
{
    // 0 - Position
    // 1 - Color
    internal class FancyFullscreenQuad : FullScreenQuad
    {
        const string Color = "color";


        public FancyFullscreenQuad()
        {
            this[Color] = StaticVBOFactory.QuadColors.Value;
            EnableAttrib(Color, 1);
        }
    }
}
