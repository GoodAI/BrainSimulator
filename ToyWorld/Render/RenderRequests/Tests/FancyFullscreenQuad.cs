using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Geometries;

namespace Render.RenderRequests.Tests
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
