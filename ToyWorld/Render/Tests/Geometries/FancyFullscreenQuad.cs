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
            Vao[Color] = QuadColors.Value;
            Vao.EnableAttrib(Color, 1);
        }
    }
}
