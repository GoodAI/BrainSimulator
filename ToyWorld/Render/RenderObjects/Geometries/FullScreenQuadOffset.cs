using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    public class FullScreenQuadOffset : FullScreenQuad
    {
        const int QuadCount = 1;

        private readonly Vector4I[] m_offsetsInternal;


        public FullScreenQuadOffset()
        {
            // We need to send the same offset to every vertex of the quad (size*4)...
            m_offsetsInternal = new Vector4I[QuadCount];

            // No init data because we update it (nearly) every step
            this[VboPosition.TextureOffsets] = new Vbo<Vector4I>(1, null, 1, hint: BufferUsageHint.StaticDraw);
            EnableAttrib(VboPosition.TextureOffsets);
        }


        public void SetTextureOffsets(params int[] data)
        {
            // We need to duplicate the data for each vertex of the quad
            Debug.Assert(QuadCount <= data.Length, "Too few data to update the texture offsets.");

            for (int i = 0; i < QuadCount; i++)
                m_offsetsInternal[i] = new Vector4I(data[i]);

            Update(VboPosition.TextureOffsets, m_offsetsInternal);
        }
    }
}
