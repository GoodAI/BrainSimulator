using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderRequests.RenderRequests;
using Render.Tests.Effects;
using Render.Tests.Geometries;
using Render.Tests.Textures;

namespace Render.Tests.RRs
{
    public interface IBasicTexRR : IRenderRequest
    { }

    class BasicTexRR : RenderRequestBase, IBasicTexRR
    {
        private FancyFullscreenQuadTex m_quad;

        private int m_offset;


        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            GL.ClearColor(Color.Black);

            m_quad = renderer.GeometryManager.Get<FancyFullscreenQuadTex>();
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            m_offset = ++m_offset % 50;

            const int tileSize = 32;
            const float tileSetWidthInv = 1 / 255f;
            const float tileSetHeightInv = 1 / 612f;

            int xIdx = m_offset % 5;
            int yIdx = m_offset / 5;

            int xPos = xIdx * tileSize;
            int yPos = yIdx * tileSize;

            float x1 = xPos * tileSetWidthInv;
            float x2 = (xPos + tileSize) * tileSetWidthInv;
            float y1 = yPos * tileSetHeightInv;
            float y2 = (yPos + tileSize) * tileSetHeightInv;


            var effect = renderer.EffectManager.Get<NoEffectTex>();
            renderer.EffectManager.Use(effect);
            effect.SetUniform(0, 0);

            var tex = renderer.TextureManager.Get<TilesetTexture>();
            renderer.TextureManager.Bind(tex);
            m_quad.SetTexCoods(new[]
            {
                0, 0f,
                0, 1,
                1, 1,
                1, 0,
                //x1, y1,
                //x2, y1,
                //x2, y2,
                //x1, y2,
            });

            m_quad.Draw();
        }

        #endregion
    }
}
