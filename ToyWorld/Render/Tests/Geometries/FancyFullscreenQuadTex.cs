using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;
using Render.RenderObjects.Geometries;

namespace Render.Tests.Geometries
{
    // 0 - Position
    // 1 - TexCoods
    internal class FancyFullscreenQuadTex : FullScreenQuad
    {
        const string Cood = "cood0";


        public FancyFullscreenQuadTex(int count)
        {
            Vao.VBOs.Add(Cood, new VBO(count, null, 2));
            Vao.EnableVBO(Cood, 1);
        }


        public void SetTexCoods(float[] data)
        {
            Vao.VBOs[Cood].Update(data);
        }
    }
}
