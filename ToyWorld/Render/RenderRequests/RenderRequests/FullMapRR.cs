using System;
using System.Drawing;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Render.Tests.Effects;
using Render.Tests.Textures;
using VRageMath;
using World.ToyWorldCore;
using Color = System.Drawing.Color;

namespace Render.RenderRequests.RenderRequests
{
    internal class FullMapRR : RenderRequestBase, IFullMapRR
    {
        private Vector2I m_size { get { return new Vector2I(Size.Width, Size.Height); } }
        private int[] m_buffer;

        private NoEffectOffset m_effect;
        private TilesetTexture m_tex;
        private FullScreenGrid m_grid;

        private Matrix m_view;
        private int m_viewPos;


        public override void Dispose()
        {
            m_effect.Dispose();
            m_tex.Dispose();
            m_grid.Dispose();
            base.Dispose();
        }


        #region IFullMapRR overrides

        public float Rotation { get; set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            GL.ClearColor(Color.DimGray);
            GL.Enable(EnableCap.Blend);
            GL.BlendEquation(BlendEquationMode.FuncAdd);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

            // Set up tileset textures
            m_tex = renderer.TextureManager.Get<TilesetTexture>();

            // Set up tile grid shaders
            m_effect = renderer.EffectManager.Get<NoEffectOffset>();
            renderer.EffectManager.Use(m_effect); // Need to use the effect to set uniforms
            m_effect.SetUniform1(m_effect.GetUniformLocation("tex"), 0);

            // Set up static uniforms
            Vector2 fullTileSize = world.TilesetTable.TileSize + world.TilesetTable.TileMargins;
            Vector2 tileCount = m_tex.Size / fullTileSize;
            m_effect.SetUniform3(m_effect.GetUniformLocation("texSizeCount"), new Vector3I(m_tex.Size.X, m_tex.Size.Y, (int)tileCount.X));
            m_effect.SetUniform4(m_effect.GetUniformLocation("tileSizeMargin"), new Vector4I(world.TilesetTable.TileSize, world.TilesetTable.TileMargins));
            m_viewPos = m_effect.GetUniformLocation("view");

            Size = new Size(world.Size.X, world.Size.Y);

            // Set up tile grid geometry
            m_buffer = new int[world.Size.Size()];
            m_grid = renderer.GeometryManager.Get<FullScreenGrid>(m_size);

        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            renderer.EffectManager.Use(m_effect);
            renderer.TextureManager.Bind(m_tex);

            if (Rotation > 0)
            {
                Matrix.CreateRotationZ(Rotation, out m_view);
                m_effect.SetUniformMatrix4(m_viewPos, m_view);
            }

            foreach (var tileLayer in world.Atlas.TileLayers)
            {
                tileLayer.GetRectangle(0, 0, world.Size.X, world.Size.Y, m_buffer);
                m_grid.SetTextureOffsets(m_buffer);

                m_grid.Draw();
            }
        }

        #endregion

    }
}
