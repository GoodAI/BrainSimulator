using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderObjects.Geometries;
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
        private FullscreenQuadTex m_quad;

        private int m_offset;


        public override void Dispose()
        {
            m_quad.Dispose();
            base.Dispose();
        }


        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            GL.ClearColor(Color.Black);

            m_quad = renderer.GeometryManager.Get<FullscreenQuadTex>();
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            m_offset = ++m_offset % 50;

            const int tileSize = 67;
            const int margin = 1;
            const float tileSetWidth = 255f;
            const float tileSetHeight = 612f;
            const float tileSetWidthInv = 1 / tileSetWidth;
            const float tileSetHeightInv = 1 / tileSetHeight;

            int xOff = m_offset % 5;
            int yOff = m_offset / 5;

            int xPos = xOff * (tileSize + margin);
            int yPos = yOff * (tileSize + margin);

            float x1 = xPos * tileSetWidthInv;
            float x2 = (xPos + tileSize) * tileSetWidthInv;
            float y2 = yPos * tileSetHeightInv;
            float y1 = (yPos + tileSize) * tileSetHeightInv;


            var effect = renderer.EffectManager.Get<NoEffectTex>();
            renderer.EffectManager.Use(effect);
            effect.SetUniform(effect.GetUniformLocation("tex"), 0);

            var tex = renderer.TextureManager.Get<TilesetTexture>();
            renderer.TextureManager.Bind(tex);
            m_quad.SetTexCoods(new float[]
            {
                x1, y1,
                x2, y1,
                x2, y2,
                x1, y2,
         
                0, 0f,
                1, 0,
                1, 1,
                0, 1,

                0, 0,
                0, tileSize,
                tileSize, tileSize,
                tileSize, 0,
   });

            m_quad.Draw();
        }

        #endregion
    }
}
