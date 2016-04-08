using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Buffers;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    internal class FullScreenGrid : GeometryBase
    {
        public Vector2I Dimensions { get; private set; }


        public FullScreenGrid(Vector2I dimensions)
        {
            Dimensions = dimensions;

            this[VboPosition.Vertices] = StaticVboFactory.GetGridVertices(dimensions);
            EnableAttrib(VboPosition.Vertices);

            this[VboPosition.TextureOffsets] = new Vbo<int>(dimensions.Size(), null, 1);
            EnableAttrib(VboPosition.TextureOffsets);
        }


        public void SetTextureOffsets(int[] data)
        {
            Update(VboPosition.TextureOffsets, data);
        }

        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, Dimensions.Size());
        }
    }
}
