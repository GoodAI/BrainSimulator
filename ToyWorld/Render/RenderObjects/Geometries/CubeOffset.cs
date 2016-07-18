using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    public class CubeOffset : Cube
    {
        const int QuadCount = 2;

        private readonly Vector4I[] m_offsetsInternal;


        public CubeOffset()
        {
            // We need to send the same offset to every vertex of the quad (size*4)...
            m_offsetsInternal = new Vector4I[QuadCount];

            // No init data because we update it (nearly) every step
            this[VboPosition.TextureOffsets] = new Vbo<Vector4I>(1, null, 1, hint: BufferUsageHint.StaticDraw);
            EnableAttrib(VboPosition.TextureOffsets);
        }


        public void SetTextureOffsets(int data)
        {
            // We need to duplicate the data for each vertex of the quad
            for (int i = 0; i < QuadCount; i++)
                m_offsetsInternal[i] = new Vector4I(data);

            Update(VboPosition.TextureOffsets, m_offsetsInternal);
        }
    }
}
