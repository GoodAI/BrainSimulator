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
using RectangleF = VRageMath.RectangleF;

namespace Render.RenderRequests
{
    public abstract class RenderRequest : IRenderRequestBase, IDisposable
    {
        private NoEffectOffset m_effect;
        private TilesetTexture m_tex;
        private FullScreenGrid m_grid;
        private FullScreenQuadOffset m_quad;

        private Matrix m_projMatrix;
        private Matrix m_viewProjectionMatrix;
        private int m_mvpPos;


        public RenderRequest()
        {
            Size = new Size(3, 3);
            Resolution = new Size(1024, 1024);
        }

        public virtual void Dispose()
        {
            m_effect.Dispose();
            m_tex.Dispose();
            m_grid.Dispose();
            m_quad.Dispose();
        }


        #region IRenderRequestBase overrides

        public PointF PositionCenter { get; protected set; }
        public virtual SizeF Size { get; protected set; }

        public System.Drawing.RectangleF View
        {
            get
            {
                return new System.Drawing.RectangleF(
                    PositionCenter.X - Size.Width / 2,
                    PositionCenter.Y - Size.Height / 2,
                    Size.Width, Size.Height);
            }
        }

        public virtual Size Resolution { get; set; }

        protected Vector2 SizeV { get { return (Vector2)Size; } set { Size = new SizeF(value.X, value.Y); } }
        protected Vector3 PositionCenterV { get { return new Vector3((Vector2)PositionCenter, 0); } set { PositionCenter = new PointF(value.X, value.Y); } }
        protected RectangleF ViewV { get { return (RectangleF)View; } }

        #endregion


        protected Matrix GetViewMatrix(Vector3 rotation, Vector3 translation, float zoom = 1)
        {
            var viewMatrix = Matrix.Identity;

            if (rotation.X > 0)
                viewMatrix = Matrix.CreateRotationZ(rotation.X);

            if (rotation.Y > 0)
                viewMatrix *= Matrix.CreateRotationX(rotation.Y);

            if (rotation.Z > 0)
                viewMatrix *= Matrix.CreateRotationY(rotation.Z);

            Vector3 tar = new Vector3(translation.X, translation.Y, 0);
            translation.Z = 20 / MathHelper.Clamp(zoom, 0.1f, 10f);
            viewMatrix *= Matrix.CreateLookAt(translation, tar, Vector3.Up);

            return viewMatrix;
        }


        public virtual void Init(RendererBase renderer, ToyWorld world)
        {
            GL.ClearColor(System.Drawing.Color.DimGray);
            GL.Enable(EnableCap.Blend);
            GL.BlendEquation(BlendEquationMode.FuncAdd);
            GL.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

            // Set up tileset textures
            m_tex = renderer.TextureManager.Get<TilesetTexture>(world.TilesetTable.GetTilesetImages());

            // Set up tile grid shaders
            m_effect = renderer.EffectManager.Get<NoEffectOffset>();
            renderer.EffectManager.Use(m_effect); // Need to use the effect to set uniforms
            m_effect.SetUniform1(m_effect.GetUniformLocation("tex"), 0);

            // Set up static uniforms
            Vector2I fullTileSize = world.TilesetTable.TileSize + world.TilesetTable.TileMargins;
            Vector2 tileCount = (Vector2)m_tex.Size / (Vector2)fullTileSize;
            m_effect.SetUniform3(m_effect.GetUniformLocation("texSizeCount"), new Vector3I(m_tex.Size.X, m_tex.Size.Y, (int)tileCount.X));
            m_effect.SetUniform4(m_effect.GetUniformLocation("tileSizeMargin"), new Vector4I(world.TilesetTable.TileSize, world.TilesetTable.TileMargins));
            m_mvpPos = m_effect.GetUniformLocation("mvp");

            // Set up tile grid geometry
            // TODO: floatovy grid, musi byt o 1 vetsi, posouvat s kamerou, preskakovat na hrane
            m_grid = renderer.GeometryManager.Get<FullScreenGrid>((Vector2I)SizeV);
            m_quad = renderer.GeometryManager.Get<FullScreenQuadOffset>();

            // View matrix is computed each frame
            m_projMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, -1, 20);
        }

        public virtual void Draw(RendererBase renderer, ToyWorld world)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            // Bind stuff to GL
            renderer.EffectManager.Use(m_effect);
            renderer.TextureManager.Bind(m_tex);

            // Setup params
            Vector2 halfWorldSize = new Vector2(world.Size.X, world.Size.Y) * 0.5f;


            // Set up transformation to screen space for tiles
            Matrix transform = Matrix.Identity;
            // Model transform -- scale to WorldSize, center on origin
            transform *= Matrix.CreateScale(halfWorldSize);
            // World transform -- move lower-left corner to origin
            transform *= Matrix.CreateTranslation(halfWorldSize);
            // View and proj transforms
            m_viewProjectionMatrix = GetViewMatrix(Vector3.Zero, PositionCenterV);
            m_viewProjectionMatrix *= m_projMatrix;
            m_effect.SetUniformMatrix4(m_mvpPos, transform * m_viewProjectionMatrix);


            // Draw tile layers
            foreach (var tileLayer in world.Atlas.TileLayers)
            {
                m_grid.SetTextureOffsets(tileLayer.GetRectangle(ViewV.Position, ViewV.Size));
                m_grid.Draw();
            }


            // Draw objects
            foreach (var objectLayer in world.Atlas.ObjectLayers)
            {
                // TODO: Setup for this object layer
                foreach (var gameObject in objectLayer.GetGameObjects(ViewV))
                {
                    // Set up transformation to screen space for the gameObject
                    transform = Matrix.Identity;
                    // Model transform
                    transform *= Matrix.CreateRotationZ(0.5f);
                    transform *= Matrix.CreateScale(gameObject.Size);
                    // World transform
                    transform *= Matrix.CreateTranslation(new Vector3(gameObject.Position, 0.01f));
                    m_effect.SetUniformMatrix4(m_mvpPos, transform * m_viewProjectionMatrix);

                    // Setup dynamic data
                    m_quad.SetTextureOffsets(gameObject.TilesetId);

                    m_quad.Draw();
                }
            }
        }
    }
}
