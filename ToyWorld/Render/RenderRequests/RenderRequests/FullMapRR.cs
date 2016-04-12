using System.Drawing;
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
using World.GameActors.Tiles;
using World.ToyWorldCore;
using Color = System.Drawing.Color;

namespace Render.RenderRequests.RenderRequests
{
    internal class FullMapRR : RenderRequestBase, IFullMapRR
    {
        private Vector2I m_size { get { return new Vector2I(Size.Width, Size.Height); } }

        private NoEffectOffset m_effect;
        private TilesetTexture m_tex;
        private FullScreenGrid m_grid;


        public override void Dispose()
        {
            m_effect.Dispose();
            m_tex.Dispose();
            m_grid.Dispose();
            base.Dispose();
        }

        #region IFullMapRR overrides

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            GL.ClearColor(Color.DimGray);
            GL.Enable(EnableCap.Blend);
            GL.BlendEquation(BlendEquationMode.FuncAdd);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

            m_tex = renderer.TextureManager.Get<TilesetTexture>();

            m_effect = renderer.EffectManager.Get<NoEffectOffset>();
            renderer.EffectManager.Use(m_effect);
            m_effect.SetUniform1(m_effect.GetUniformLocation("tex"), 0);

            Vector2 fullTileSize = world.TilesetTable.TileSize + world.TilesetTable.TileMargins;
            Vector2 tileCount = m_tex.Size / fullTileSize;
            m_effect.SetUniform3(m_effect.GetUniformLocation("texSizeCount"), new Vector3I(m_tex.Size.X, m_tex.Size.Y, (int)tileCount.X));
            m_effect.SetUniform4(m_effect.GetUniformLocation("tileSizeMargin"), new Vector4I(world.TilesetTable.TileSize, world.TilesetTable.TileMargins));

            //Size = new Size((int)tileCount.X, (int)tileCount.Y / 2);
            Size = new Size(world.Size.X, world.Size.Y);

            m_grid = renderer.GeometryManager.Get<FullScreenGrid>(m_size);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            renderer.EffectManager.Use(m_effect);
            renderer.TextureManager.Bind(m_tex);

            var layer = world.Atlas.TileLayers.First();
            var rect = layer.GetRectangle(0, 0, world.Size.X, world.Size.Y);

            //int[] offsets = Enumerable.Range(0, m_size.Size()).ToArray();
            int[] offsets = rect.Cast<Tile>().Select(t => t != null ? t.TileType : -1).ToArray();
            m_grid.SetTextureOffsets(offsets);

            m_grid.Draw();
        }

        #endregion
    }
}
