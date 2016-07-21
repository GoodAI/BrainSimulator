using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    public class GridOffset : Grid
    {
        public GridOffset(Vector2I dimensions)
            : base(dimensions)
        {
            this[VboPosition.TextureOffsets] = new Vbo<Vector4I>(dimensions.Size(), null, 1);
            EnableAttribI(VboPosition.TextureOffsets, type: VertexAttribIntegerType.UnsignedShort);
        }


        public int GetPaddedBufferSize()
        {
            return Dimensions.Size();
        }

        public void GetPaddedTextureOffsets(ushort[] data, Vector4I[] paddedData)
        {
            // We need to send the same offset to every vertex of the quad (size*4)...
            int size = Dimensions.Size();
            Debug.Assert(size <= data.Length, "Too few data to update the texture offsets.");
            Debug.Assert(size <= paddedData.Length, "Too few data to update the texture offsets.");

            for (int i = 0; i < size; i++)
                paddedData[i] = new Vector4I((data[i] << 16) | data[i]);
        }

        public void SetTextureOffsets(Vector4I[] data)
        {
            Update(VboPosition.TextureOffsets, data);
        }
    }
}
