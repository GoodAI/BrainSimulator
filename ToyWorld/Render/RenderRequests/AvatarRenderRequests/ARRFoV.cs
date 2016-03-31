using System;
using System.Diagnostics;
using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Geometries.Buffers;
using Render.Renderer;

namespace Render.RenderRequests.AvatarRenderRequests
{
    internal class ARRFoV : AvatarRenderRequestBase, IAvatarRenderRequestFoV
    {
        //VBO m_pbo;
        private bool m_odd;


        internal ARRFoV(int avatarID)
            : base(avatarID)
        { }


        #region IAvatarRenderRequestFoV overrides

        public uint[] Image { get; protected set; }

        #endregion

        #region AvatarRenderRequestBase overrides

        public override float Size { get { return Image.Length; } }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(IRenderer renderer)
        {
            //m_pbo = new VBO(renderer.Window.Width * renderer.Window.Height, target: BufferTarget.PixelPackBuffer, hint: BufferUsageHint.StreamRead);
            Image = new uint[renderer.Window.Width * renderer.Window.Height];

            GL.ClearColor(Color.Black);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            GL.Begin(PrimitiveType.Lines);

            if (m_odd = !m_odd)
            {
                GL.Color3(Color.Red);
                GL.Vertex3(0, 0, 0);
                GL.Vertex3(1, 1, 1);
            }
            else
            {
                GL.Color3(Color.Green);
                GL.Vertex3(0, 0, 0);
                GL.Vertex3(-1, -1, -1);
            }

            GL.End();

            //m_pbo.Bind();
            //GL.ReadPixels(0,0,renderer.Window.Width, renderer.Window.Height,PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);
            //m_pbo.Unbind();

            GL.ReadPixels(0, 0, renderer.Window.Width, renderer.Window.Height, PixelFormat.Rgba, PixelType.UnsignedByte, Image);
        }

        #endregion
    }
}
