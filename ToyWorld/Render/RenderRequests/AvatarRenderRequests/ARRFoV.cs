using System;
using System.Diagnostics;
using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.Tests.Effects;
using Render.Tests.Geometries;

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

        public override void Init(RendererBase renderer)
        {
            //m_pbo = new VBO(renderer.Window.Width * renderer.Window.Height, target: BufferTarget.PixelPackBuffer, hint: BufferUsageHint.StreamRead);

            // TODO: mel by mit vlastni rendertarget s custom dims, spravovanej nejakym managerem
            Image = new uint[renderer.Width * renderer.Height];

            GL.ClearColor(Color.Black);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            renderer.EffectManager.Use<NoEffect>();
            renderer.GeometryManager.Draw<FancyFullscreenQuad>();

            GL.ReadPixels(0, 0, renderer.Width, renderer.Height, PixelFormat.Rgba, PixelType.UnsignedByte, Image);
        }

        #endregion
    }
}
