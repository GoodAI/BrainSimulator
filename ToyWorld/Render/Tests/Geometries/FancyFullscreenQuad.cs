using Render.RenderObjects.Geometries;

namespace Render.Tests.Geometries
{
    // 0 - Position
    // 1 - Color
    internal class FancyFullscreenQuad : FullScreenQuad
    {
        public FancyFullscreenQuad()
        {
            Vao.AddVBO(QuadColors.Value, 1);
        }
    }
}
