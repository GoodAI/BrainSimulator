using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    public class FullScreenGridOffset : FullScreenGrid
    {
        private readonly Vector4I[] m_offsetsInternal;


        public FullScreenGridOffset(Vector2I dimensions)
            : base(dimensions)
        {
            // We need to send the same offset to every vertex of the quad (size*4)...
            m_offsetsInternal = new Vector4I[dimensions.Size()];

            this[VboPosition.TextureOffsets] = new Vbo<Vector4I>(dimensions.Size(), null, 1);
            EnableAttrib(VboPosition.TextureOffsets);
        }


        public void SetTextureOffsets(int[] data)
        {
            // We need to duplicate the data for each vertex of the quad
            int size = Dimensions.Size();
            Debug.Assert(size <= data.Length, "Too few data to update the texture offsets.");

            for (int i = 0; i < size; i++)
                m_offsetsInternal[i] = new Vector4I(data[i] - 1);

            Update(VboPosition.TextureOffsets, m_offsetsInternal);
        }
    }
}
