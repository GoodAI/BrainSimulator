using System;
using System.Diagnostics;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderObjects.Effects;
using Render.RenderObjects.Geometries;
using Render.RenderObjects.Textures;
using VRageMath;
using World.Physics;
using World.ToyWorldCore;
using Rectangle = VRageMath.Rectangle;
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


        protected Vector2 PositionCenterV { get; set; }
        protected Vector2 SizeV { get; set; }
        protected RectangleF ViewV { get { return new RectangleF(Vector2.Zero, SizeV) { Center = PositionCenterV }; } }

        private Rectangle GridView
        {
            get
            {
                var rect = new RectangleF(Vector2.Zero, ViewV.Size + 2) { Center = ViewV.Center };
                return new Rectangle(
                    new Vector2I(
                        (int)Math.Ceiling(rect.Position.X),
                        (int)Math.Ceiling(rect.Position.Y)),
                    (Vector2I)rect.Size);
            }
        }


        public RenderRequest()
        {
            SizeV = new Vector2(3, 3);
            Resolution = new System.Drawing.Size(1024, 1024);
        }

        public virtual void Dispose()
        {
            m_effect.Dispose();
            m_tex.Dispose();
            m_grid.Dispose();
            m_quad.Dispose();
        }


        #region IRenderRequestBase overrides

        public System.Drawing.PointF PositionCenter
        {
            get { return new System.Drawing.PointF(PositionCenterV.X, PositionCenterV.Y); }
            protected set { PositionCenterV = new Vector2(value.X, value.Y); }
        }

        public System.Drawing.SizeF Size
        {
            get { return new System.Drawing.SizeF(SizeV.X, SizeV.Y); }
            protected set { SizeV = new Vector2(value.Width, value.Height); }
        }

        public System.Drawing.RectangleF View { get { return new System.Drawing.RectangleF(PositionCenter, Size); } }

        public virtual System.Drawing.Size Resolution { get; set; }

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
            const int baseIntensity = 50;
            GL.ClearColor(System.Drawing.Color.FromArgb(baseIntensity, baseIntensity, baseIntensity));
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
            m_grid = renderer.GeometryManager.Get<FullScreenGrid>(GridView.Size);
            m_quad = renderer.GeometryManager.Get<FullScreenQuadOffset>();

            // View matrix is computed each frame
            m_projMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, -1, 20);
        }

        public virtual void Draw(RendererBase renderer, ToyWorld world)
        {
            if (GridView.Size != m_grid.Dimensions)
                m_grid = renderer.GeometryManager.Get<FullScreenGrid>(GridView.Size);


            GL.Clear(ClearBufferMask.ColorBufferBit);

            // Bind stuff to GL
            renderer.EffectManager.Use(m_effect);
            renderer.TextureManager.Bind(m_tex);


            // Set up transformation to screen space for tiles
            Matrix transform = Matrix.Identity;
            // Model transform -- scale from (-1,1) to viewSize/2, center on origin
            transform *= Matrix.CreateScale((Vector2)GridView.Size / 2);
            // World transform -- move center to view center
            transform *= Matrix.CreateTranslation(new Vector3(GridView.Center));
            // View and proj transforms
            m_viewProjectionMatrix = GetViewMatrix(Vector3.Zero, new Vector3(PositionCenterV));
            m_viewProjectionMatrix *= m_projMatrix;
            m_effect.SetUniformMatrix4(m_mvpPos, transform * m_viewProjectionMatrix);


            // Draw tile layers
            foreach (var tileLayer in world.Atlas.TileLayers)
            {
                m_grid.SetTextureOffsets(tileLayer.GetRectangle(GridView));
                m_grid.Draw();
            }


            // Draw objects
            foreach (var objectLayer in world.Atlas.ObjectLayers)
            {
                // TODO: Setup for this object layer
                foreach (var gameObject in objectLayer.GetGameObjects(new RectangleF(GridView)))
                {
                    // Set up transformation to screen space for the gameObject
                    transform = Matrix.Identity;
                    // Model transform
                    IDirectable dir = gameObject as IDirectable;
                    if (dir != null)
                        transform *= Matrix.CreateRotationZ(dir.Direction);
                    transform *= Matrix.CreateScale(gameObject.Size * 0.5f); // from (-1,1) to (-size,size)/2
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
