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
            Vao.VBOs[Color] = QuadColors.Value;
            Vao.EnableVBO(Color, 1);
        }
    }
}
