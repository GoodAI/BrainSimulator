using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderRequests.AvatarRenderRequests;
using Render.Tests.Effects;
using Render.Tests.Geometries;

namespace Render.Tests.RRs
{
    public interface IBasicARR : IAvatarRenderRequest
    { }

    class BasicARR : AvatarRenderRequestBase, IBasicARR
    {
        public BasicARR(int avatarID)
            : base(avatarID)
        { }


        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            GL.ClearColor(Color.Black);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            renderer.EffectManager.Use<NoEffect>();
            renderer.GeometryManager.Draw<FancyFullscreenQuad>();
        }

        #endregion
    }
}
