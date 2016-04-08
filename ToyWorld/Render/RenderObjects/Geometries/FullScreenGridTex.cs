using Render.RenderObjects.Buffers;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    internal class FullScreenGridTex : GeometryBase
    {
        public FullScreenGridTex(Vector2I dimensions)
        {
            this[VboPosition.Vertices] = StaticVboFactory.GetGridVertices(dimensions);
            EnableAttrib(VboPosition.Vertices);

            this[VboPosition.TextureOffsets] = new Vbo<int>(dimensions.Size(), null, 1);
            EnableAttrib(VboPosition.TextureOffsets);
        }


        public void SetTexCoods(float[] data)
        {
            Update(VboPosition.TextureOffsets, data);
        }

        public override void Draw()
        {
        }
    }
}
