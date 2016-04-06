using Render.Geometries.Buffers;

namespace Render.RenderObjects.Geometries
{
    // 0 - Position
    // 1 - TexCoods
    internal class FullscreenQuadTex : FullScreenQuad
    {
        const string Cood = "cood0";


        public FullscreenQuadTex()
        {
            Vao[Cood] = new VBO(8, null, 2);
            Vao.EnableAttrib(Cood, 1);
        }


        public void SetTexCoods(float[] data)
        {
            Vao[Cood].Update(data);
        }
    }
}
