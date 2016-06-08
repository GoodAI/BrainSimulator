using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Effects;
using RenderingBase.Renderer;
using RenderingBase.RenderObjects.Textures;
using TmxMapSerializer.Elements;
using VRageMath;
using World.GameActors.GameObjects;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class OverlayRenderer
        : RRRendererBase<OverlaySettings>, IDisposable
    {
        #region Fields

        protected const TextureUnit UIOverlayTextureBindPosition = TextureUnit.Texture5;

        protected NoEffectOffset m_overlayEffect;
        protected TilesetTexture m_overlayTexture;

        #endregion

        #region Genesis

        public virtual void Dispose()
        {
            if (m_overlayEffect != null)
                m_overlayEffect.Dispose();
            if (m_overlayTexture != null)
                m_overlayTexture.Dispose();
        }

        #endregion

        #region Init

        public virtual void Init(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world, OverlaySettings settings)
        {
            // Set up overlay textures
            IEnumerable<Tileset> tilesets = world.TilesetTable.GetOverlayImages();
            TilesetImage[] tilesetImages = tilesets.Select(t =>
                    new TilesetImage(
                        t.Image.Source,
                        new Vector2I(t.Tilewidth, t.Tileheight),
                        new Vector2I(t.Spacing),
                        world.TilesetTable.TileBorder))
                .ToArray();

            m_overlayTexture = renderer.TextureManager.Get<TilesetTexture>(tilesetImages);
            renderer.TextureManager.Bind(m_overlayTexture, UIOverlayTextureBindPosition);

            // Set up overlay shader
            m_overlayEffect = renderer.EffectManager.Get<NoEffectOffset>();
            renderer.EffectManager.Use(m_overlayEffect); // Need to use the effect to set uniforms

            // Set up static uniforms
            Vector2I tileSize = tilesetImages[0].TileSize;
            Vector2I tileMargins = tilesetImages[0].TileMargin;
            Vector2I tileBorder = tilesetImages[0].TileBorder;

            Vector2I fullTileSize = tileSize + tileMargins + tileBorder * 2; // twice the border, on each side once
            Vector2 tileCount = (Vector2)m_overlayTexture.Size / (Vector2)fullTileSize;
            m_overlayEffect.TexSizeCountUniform(new Vector3I(m_overlayTexture.Size.X, m_overlayTexture.Size.Y, (int)tileCount.X));
            m_overlayEffect.TileSizeMarginUniform(new Vector4I(tileSize, tileMargins));
            m_overlayEffect.TileBorderUniform(tileBorder);

            m_overlayEffect.AmbientUniform(new Vector4(1, 1, 1, 1));
        }

        #endregion

        #region Draw

        public virtual void Draw(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            m_frontFbo.Bind();

            // Bind stuff to GL
            renderer.TextureManager.Bind(m_tilesetTexture);
            renderer.EffectManager.Use(m_effect);
            m_effect.TextureUniform(0);

            DrawOverlays(renderer, world);
        }

        protected void DrawAvatarTool(RendererBase<ToyWorld> renderer, IAvatar avatar, Vector2 size, Vector2 position, ToolBackgroundType type = ToolBackgroundType.BrownBorder)
        {
            if (FlipYAxis)
            {
                size.Y = -size.Y;
                position.Y = -position.Y;
            }

            Matrix transform = Matrix.CreateScale(size);
            transform *= Matrix.CreateTranslation(position.X, position.Y, 0.01f);


            // Draw the inventory background
            renderer.TextureManager.Bind(m_overlayTexture, UIOverlayTextureBindPosition);
            renderer.EffectManager.Use(m_overlayEffect);
            m_overlayEffect.TextureUniform((int)UIOverlayTextureBindPosition - (int)TextureUnit.Texture0);
            m_overlayEffect.ModelViewProjectionUniform(ref transform);

            m_quadOffset.SetTextureOffsets((int)type);
            m_quadOffset.Draw();


            // Draw the inventory Tool
            if (avatar.Tool != null)
            {
                renderer.TextureManager.Bind(m_tilesetTexture);
                renderer.EffectManager.Use(m_effect);
                m_effect.DiffuseUniform(new Vector4(1, 1, 1, 1));

                Matrix toolTransform = Matrix.CreateScale(0.7f) * transform;
                m_effect.ModelViewProjectionUniform(ref toolTransform);

                m_quadOffset.SetTextureOffsets(avatar.Tool.TilesetId);
                m_quadOffset.Draw();
            }
        }

        #endregion
    }
}
