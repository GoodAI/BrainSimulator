using System;
using System.Diagnostics;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Buffers;
using VRageMath;

namespace Render.RenderObjects.Geometries
{
    internal class FullScreenGrid : GeometryBase
    {
        private readonly Vector4I[] m_offsetsInternal;

        public Vector2I Dimensions { get; private set; }


        public FullScreenGrid(Vector2I dimensions)
        {
            Dimensions = dimensions;

            this[VboPosition.Vertices] = StaticVboFactory.GetGridVertices(dimensions);
            EnableAttrib(VboPosition.Vertices);

            // We need to send the same offset to every vertex of the quad (size*4)...
            m_offsetsInternal = new Vector4I[dimensions.Size()];

            this[VboPosition.TextureOffsets] = new Vbo<Vector4I>(dimensions.Size(), null, 1);
            EnableAttrib(VboPosition.TextureOffsets);
        }


        public void SetTextureOffsets(int[] data)
        {
            // We need to duplicate the data for each vertex of the quad
            int size = Dimensions.Size();
            Debug.Assert(size <= data.Length, "Too few data to update the tex offsets.");

            for (int i = 0; i < size; i++)
                m_offsetsInternal[i] = new Vector4I(data[i]);

            Update(VboPosition.TextureOffsets, m_offsetsInternal);
        }

        public override void Draw()
        {
            GL.BindVertexArray(Handle);
            GL.DrawArrays(PrimitiveType.Quads, 0, Dimensions.Size() * 4);
        }
    }
}
