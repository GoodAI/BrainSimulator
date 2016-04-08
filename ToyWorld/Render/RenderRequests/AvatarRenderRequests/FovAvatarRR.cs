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
    internal class FovAvatarRR : AvatarRRBase, IFovAvatarRR
    {
        internal FovAvatarRR(int avatarID)
            : base(avatarID)
        { }


        #region IFovAvatarRR overrides

        public uint[] Image { get; private set; }

        #endregion

        #region AvatarRRBase overrides

        public override float Size { get; set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            //m_pbo = new Vbo<T>(renderer.Window.Width * renderer.Window.Height, target: BufferTarget.PixelPackBuffer, hint: BufferUsageHint.StreamRead);

            // TODO: mel by mit vlastni rendertarget s custom dims, spravovanej nejakym managerem
            Image = new uint[renderer.Width * renderer.Height];

            GL.ClearColor(Color.Black);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            var effect = renderer.EffectManager.Get<NoEffect>();
            renderer.EffectManager.Use(effect);
            renderer.GeometryManager.Get<FancyFullscreenQuad>().Draw();

            GL.ReadPixels(0, 0, renderer.Width, renderer.Height, PixelFormat.Rgba, PixelType.UnsignedByte, Image);
        }

        #endregion
    }
}
