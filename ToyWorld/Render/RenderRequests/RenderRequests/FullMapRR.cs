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
        readonly Vector2I m_mapDims = new Vector2I(3);
        readonly Vector4I m_tileSizeMargin = new Vector4I(16, 16, 1, 1);

        private NoEffectOffset m_effect;
        private TilesetTexture m_tex;
        private FullScreenGrid m_grid;


        #region IFullMapRR overrides

        #endregion

        #region AvatarRenderRequestBase overrides

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer)
        {
            GL.ClearColor(Color.Black);

            m_tex = renderer.TextureManager.Get<TilesetTexture>();

            m_effect = renderer.EffectManager.Get<NoEffectOffset>();
            m_effect.SetUniform1(m_effect.GetUniformLocation("tex"), 0);

            Vector2 fullTileSize = new Vector2I(m_tileSizeMargin.X + m_tileSizeMargin.Z, m_tileSizeMargin.Y + m_tileSizeMargin.W);
            Vector2 tileCount = m_tex.Size / fullTileSize;
            Vector2I texSizeCount = new Vector2I((int)tileCount.X, (int)tileCount.Y);
            m_effect.SetUniform4(m_effect.GetUniformLocation("texSizeCount"), new Vector4I(m_tex.Size.X, m_tex.Size.Y, texSizeCount.X, texSizeCount.Y));
            m_effect.SetUniform4(m_effect.GetUniformLocation("tileSizeMargin"), m_tileSizeMargin);

            m_grid = renderer.GeometryManager.Get<FullScreenGrid>(m_mapDims);
        }

        public override void Draw(RendererBase renderer)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            renderer.EffectManager.Use(m_effect);
            renderer.TextureManager.Bind(m_tex);

            int[] offsets = Enumerable.Range(0, m_mapDims.Size()).ToArray();
            m_grid.SetTextureOffsets(offsets);

            m_grid.Draw();
        }

        #endregion
    }
}
