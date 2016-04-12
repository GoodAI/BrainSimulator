using System;
using System.Diagnostics;
using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.Tests.Effects;
using Render.Tests.Geometries;
using World.ToyWorldCore;

namespace Render.RenderRequests.AvatarRenderRequests
{
    internal class FovAvatarRR : AvatarRRBase, IFovAvatarRR
    {
        private NoEffect m_effect;
        private FancyFullscreenQuad m_quad;


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

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            GL.ClearColor(Color.Black);

            // TODO: mel by mit vlastni rendertarget s custom dims, spravovanej nejakym managerem
            Image = new uint[renderer.Width * renderer.Height];

            m_effect = renderer.EffectManager.Get<NoEffect>();
            m_quad = renderer.GeometryManager.Get<FancyFullscreenQuad>();

            //m_pbo = new Vbo<T>(renderer.Window.Width * renderer.Window.Height, target: BufferTarget.PixelPackBuffer, hint: BufferUsageHint.StreamRead);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            renderer.EffectManager.Use(m_effect);
            m_quad.Draw();

            GL.ReadPixels(0, 0, renderer.Width, renderer.Height, PixelFormat.Rgba, PixelType.UnsignedByte, Image);
        }

        #endregion
    }
}
