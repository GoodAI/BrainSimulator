using System.Linq;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Render.RenderRequests.AvatarRenderRequests;
using Render.Tests.Effects;
using Render.Tests.Geometries;
using Render.Tests.Textures;
using VRageMath;
using Color = System.Drawing.Color;

namespace Render.RenderRequests.RenderRequests
{
    internal class FullMapRR : RenderRequestBase, IFullMapRR
    {
        Vector2I m_dims = new Vector2I(3);


        #region IFullMapRR overrides

        #endregion

        #region AvatarRenderRequestBase overrides

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            GL.ClearColor(Color.Black);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            var effect = renderer.EffectManager.Get<NoEffectOffset>();
            renderer.EffectManager.Use(effect);
            effect.SetUniform(effect.GetUniformLocation("tex"), 0);

            var tex = renderer.TextureManager.Get<TilesetTexture>();
            renderer.TextureManager.Bind(tex);

            int[] offsets = Enumerable.Range(0, m_dims.Size()).ToArray();
            renderer.GeometryManager.Get<FullScreenGrid>(m_dims).SetTextureOffsets(offsets);

            renderer.GeometryManager.Get<FullScreenGrid>(m_dims).Draw();
        }

        #endregion
    }
}
