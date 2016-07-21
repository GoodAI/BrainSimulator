using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    public class CubeGridOffset : CubeGrid
    {
        const int QuadCount = 2;


        public CubeGridOffset(Vector2I dimensions)
            : base(dimensions)
        {
            this[VboPosition.TextureOffsets] = new Vbo<Vector4I>(dimensions.Size() * QuadCount, null, 1);
            EnableAttribI(VboPosition.TextureOffsets, type: VertexAttribIntegerType.UnsignedShort);
        }

        }


        public int GetPaddedBufferSize()
        {
            return Dimensions.Size() * QuadCount;
        }

        public void GetPaddedTextureOffsets(ushort[] data, Vector4I[] paddedData)
        {
            // We need to send the same offset to every vertex of the quad (size*4)...
            int size = Dimensions.Size();
            Debug.Assert(size <= data.Length, "Too few data to update the texture offsets.");
            Debug.Assert(size <= paddedData.Length / QuadCount, "Too few data to update the texture offsets.");

            for (int i = 0; i < paddedData.Length; )
            {
                int val = data[i / QuadCount];

                for (int j = 0; j < QuadCount; j++)
                    paddedData[i++] = new Vector4I(val);
            }
        }

        public void SetTextureOffsets(Vector4I[] data)
        {
            Update(VboPosition.TextureOffsets, data);
        }
    }
}
