using Render.RenderObjects.Buffers;

namespace Render.RenderObjects.Geometries
{
    // 0 - Position
    // 1 - TexCoods
    internal class FullscreenQuadTex : FullScreenQuad
    {
        const string Cood = "cood0";


        public FullscreenQuadTex()
        {
            this[Cood] = new VBO<float>(8, null, 2);
            EnableAttrib(Cood, 1);
        }


        public void SetTexCoods(float[] data)
        {
            Update(Cood, data);
        }
    }
}
