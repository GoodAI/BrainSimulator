using RenderingBase.RenderObjects.Buffers;
using VRageMath;

namespace RenderingBase.RenderObjects.Geometries
{
    internal class FullScreenQuadTex : FullScreenQuad
    {
        public FullScreenQuadTex()
        {
            // 4 vertices * 2 vector components
            // No init data because we update it (nearly) every step
            this[VboPosition.TextureCoords] = new Vbo<Vector2>(4, null, 2);
            EnableAttrib(VboPosition.TextureCoords);
        }


        public void SetTextureCoords(Vector2[] data)
        {
            Update(VboPosition.TextureCoords, data);
        }
    }
}
