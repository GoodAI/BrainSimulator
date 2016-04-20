using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    internal class FullscreenQuadColor : FullScreenQuad
    {
        public FullscreenQuadColor()
        {
            this[VboPosition.Colors] = StaticVboFactory.QuadColors;
            EnableAttrib(VboPosition.Colors);
        }
    }
}
