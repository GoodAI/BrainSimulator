using OpenTK;
using OpenTK.Graphics.OpenGL;
using System;
using System.Drawing;

namespace GoodAI.Core.Observers
{
    [Flags]
    public enum MyVertexAttrib
    {
        None = 0,
        Position = 1,
        Normal = 2,
        TexCoord = 4,
        Color = 8
    }

    public abstract class MyShape
    {
        public Vector3 Translation;
        internal MyAbstractObserver Observer { get; set; }

        public abstract void Render();
    }

    public class MyDefaultShape : MyShape
    {
        public const double SCALE = 0.1;

        public override void Render()
        {
            GL.Enable(EnableCap.Texture2D);            
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, Observer.TextureVBO);            
            GL.BindTexture(TextureTarget.Texture2D, Observer.TextureId);
            
            GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, Observer.TextureWidth, Observer.TextureHeight, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero); // h_pixels IntPtr.Zero

            GL.Color3(1.0f, 1.0f, 1.0f);

            double quadWidthHalf = Observer.TextureWidth * SCALE * 0.5;
            double quadHeightHalf = Observer.TextureHeight * SCALE * 0.5;

            GL.Begin(PrimitiveType.Quads);
            GL.TexCoord2(0, 0);
            GL.Vertex2(-quadWidthHalf, quadHeightHalf);

            GL.TexCoord2(0, 1.0f);
            GL.Vertex2(-quadWidthHalf, -quadHeightHalf); // 1 -1

            GL.TexCoord2(1.0f, 1.0f);
            GL.Vertex2(quadWidthHalf, -quadHeightHalf);

            GL.TexCoord2(1.0f, 0);
            GL.Vertex2(quadWidthHalf, quadHeightHalf); //-1 1
            GL.End();

            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);
            GL.BindTexture(TextureTarget.Texture2D, 0);
        }
    }

    public class MyBufferedPrimitive : MyShape
    {
        public int VertexOffset { get; set; }
        public int NormalOffset { get; set; }
        public int TexCoordOffset { get; set; }
        public int ColorOffset { get; set; }
        public Color BaseColor { get; set; }
        public int VertexCount { get; set; }

        private PrimitiveType m_primitive;
        private MyVertexAttrib m_vertexSpecs;

        public MyBufferedPrimitive(PrimitiveType primitive, int vertexCount, MyVertexAttrib vertexSpecs)
        {
            m_primitive = primitive;
            VertexCount = vertexCount;
            m_vertexSpecs = vertexSpecs;

            BaseColor = Color.White;
        }

        public override void Render()
        {
            GL.Color3(BaseColor);

            if ((m_vertexSpecs & MyVertexAttrib.TexCoord) > 0)
            {
                GL.Enable(EnableCap.Texture2D);
                GL.BindBuffer(BufferTarget.PixelUnpackBuffer, Observer.TextureVBO);
                GL.BindTexture(TextureTarget.Texture2D, Observer.TextureId);

                GL.TexSubImage2D(TextureTarget.Texture2D, 0, 0, 0, Observer.TextureWidth, Observer.TextureHeight, PixelFormat.Bgra, PixelType.UnsignedByte, IntPtr.Zero);
            }
            else
            {
                GL.Disable(EnableCap.Texture2D);
            }

            GL.BindBuffer(BufferTarget.ArrayBuffer, Observer.VertexVBO);

            EnableClientStates();

            GL.DrawArrays(m_primitive, 0, VertexCount);

            DisableClientStates();

            if ((m_vertexSpecs & MyVertexAttrib.TexCoord) > 0)
            {                
                GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);
                GL.BindTexture(TextureTarget.Texture2D, 0);
            }            

            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);            
        }             

        private void EnableClientStates()
        {
            if ((m_vertexSpecs & MyVertexAttrib.Position) > 0)
            {
                GL.EnableClientState(ArrayCap.VertexArray);
                GL.VertexPointer(3, VertexPointerType.Float, 0, VertexOffset * sizeof(float));
            }

            if ((m_vertexSpecs & MyVertexAttrib.Normal) > 0)
            {
                GL.EnableClientState(ArrayCap.NormalArray);
                GL.NormalPointer(NormalPointerType.Float, 0, NormalOffset * sizeof(float));
            }

            if ((m_vertexSpecs & MyVertexAttrib.TexCoord) > 0)
            {
                GL.EnableClientState(ArrayCap.TextureCoordArray);
                GL.TexCoordPointer(2, TexCoordPointerType.Float, 0, TexCoordOffset * sizeof(float));
            }

            if ((m_vertexSpecs & MyVertexAttrib.Color) > 0)
            {
                GL.EnableClientState(ArrayCap.ColorArray);
                GL.ColorPointer(3, ColorPointerType.Float, 0, ColorOffset * sizeof(float));
            }
        }

        private void DisableClientStates()
        {
            if ((m_vertexSpecs & MyVertexAttrib.Position) > 0)
            {
                GL.DisableClientState(ArrayCap.VertexArray);
            }

            if ((m_vertexSpecs & MyVertexAttrib.Normal) > 0)
            {
                GL.DisableClientState(ArrayCap.NormalArray);
            }

            if ((m_vertexSpecs & MyVertexAttrib.TexCoord) > 0)
            {
                GL.DisableClientState(ArrayCap.TextureCoordArray);
            }

            if ((m_vertexSpecs & MyVertexAttrib.Color) > 0)
            {
                GL.DisableClientState(ArrayCap.ColorArray);
            }
        }
    }
}
