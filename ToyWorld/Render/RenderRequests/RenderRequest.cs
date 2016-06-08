using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Effects;
using Render.RenderObjects.Geometries;
using RenderingBase.Renderer;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using RenderingBase.RenderObjects.Textures;
using RenderingBase.RenderRequests;
using TmxMapSerializer.Elements;
using VRageMath;
using World.Atlas.Layers;
using World.GameActors.GameObjects;
using World.Physics;
using World.ToyWorldCore;
using Rectangle = VRageMath.Rectangle;
using RectangleF = VRageMath.RectangleF;

namespace Render.RenderRequests
{
    public abstract class RenderRequest
        : IRenderRequestBaseInternal<ToyWorld>
    {
        [Flags]
        protected enum DirtyParams
        {
            None = 0,
            Size = 1,
            Resolution = 1 << 1,
            Image = 1 << 2,
            Smoke = 1 << 3,
            Noise = 1 << 4,
            Overlay = 1 << 5,
        }


        #region Fields

        protected const float AmbientTerm = 0.25f;

        protected BasicFbo m_frontFbo, m_backFbo;
        protected BasicFboMultisample m_fboMs;

        protected NoEffectOffset m_effect;

        protected TilesetTexture m_tilesetTexture;

        protected FullScreenGridOffset m_gridOffset;
        protected FullScreenQuadOffset m_quadOffset;
        protected FullScreenQuad m_quad;

        protected Matrix m_projMatrix;
        protected Matrix m_viewProjectionMatrix;


        protected EffectRenderer m_effectRenderer;
        protected PostprocessRenderer m_postprocessRenderer;
        protected OverlayRenderer m_overlayRenderer;
        protected ImageRenderer m_imageRenderer;

        #endregion

        #region Genesis

        protected RenderRequest()
        {
            PositionCenterV = new Vector3(0, 0, 20);
            SizeV = new Vector2(3, 3);
            Resolution = new System.Drawing.Size(1024, 1024);
        }

        public virtual void Dispose()
        {
            if (m_frontFbo != null)
                m_frontFbo.Dispose();
            if (m_backFbo != null)
                m_backFbo.Dispose();
            if (m_fboMs != null)
                m_fboMs.Dispose();

            m_effect.Dispose();
            m_tilesetTexture.Dispose();

            if (m_gridOffset != null) // It is initialized during Draw
                m_gridOffset.Dispose();
            m_quadOffset.Dispose();
            m_quad.Dispose();
        }

        #endregion

        #region View control properties

        /// <summary>
        /// The position of the center of view.
        /// </summary>
        protected Vector3 PositionCenterV { get; set; }
        /// <summary>
        /// The position of the center of view. Equivalent to PositionCenterV (except for the z value).
        /// </summary>
        protected Vector2 PositionCenterV2 { get { return new Vector2(PositionCenterV); } set { PositionCenterV = new Vector3(value, PositionCenterV.Z); } }

        private Vector2 m_sizeV;
        protected Vector2 SizeV
        {
            get { return m_sizeV; }
            set
            {
                const float minSize = 0.01f;
                m_sizeV = new Vector2(Math.Max(minSize, value.X), Math.Max(minSize, value.Y));
                m_dirtyParams |= DirtyParams.Size;
            }
        }

        protected virtual RectangleF ViewV { get { return new RectangleF(Vector2.Zero, SizeV) { Center = new Vector2(PositionCenterV) }; } }

        private Rectangle GridView
        {
            get
            {
                var view = ViewV;
                var positionOffset = new Vector2(view.Width % 2, view.Height % 2); // Always use a grid with even-sized sides to have it correctly centered
                var rect = new RectangleF(Vector2.Zero, view.Size + 2 + positionOffset) { Center = view.Center - positionOffset };
                return new Rectangle(
                    new Vector2I(
                        (int)Math.Ceiling(rect.Position.X),
                        (int)Math.Ceiling(rect.Position.Y)),
                    new Vector2I(rect.Size));
            }
        }

        #endregion

        #region IRenderRequestBase overrides

        public bool CopyToWindow { get; set; }

        #region View controls

        public System.Drawing.PointF PositionCenter
        {
            get { return new System.Drawing.PointF(PositionCenterV.X, PositionCenterV.Y); }
            protected set { PositionCenterV2 = new Vector2(value.X, value.Y); }
        }

        public virtual System.Drawing.SizeF Size
        {
            get { return new System.Drawing.SizeF(SizeV.X, SizeV.Y); }
            set { SizeV = (Vector2)value; }
        }

        public System.Drawing.RectangleF View
        {
            get { return new System.Drawing.RectangleF(PositionCenter, Size); }
        }

        private bool m_flipYAxis;
        public bool FlipYAxis
        {
            get { return m_flipYAxis; }
            set
            {
                m_flipYAxis = value;
                m_dirtyParams |= DirtyParams.Size;
            }
        }

        #endregion

        #region Resolution

        private System.Drawing.Size m_resolution;
        public System.Drawing.Size Resolution
        {
            get { return m_resolution; }
            set
            {
                const int minResolution = 16;
                const int maxResolution = 4096;
                if (value.Width < minResolution || value.Height < minResolution)
                    throw new ArgumentOutOfRangeException("value", "Invalid resolution: must be greater than " + minResolution + " pixels.");
                if (value.Width > maxResolution || value.Height > maxResolution)
                    throw new ArgumentOutOfRangeException("value", "Invalid resolution: must be at most " + maxResolution + " pixels.");

                m_resolution = value;
                m_dirtyParams |= DirtyParams.Resolution | DirtyParams.Image;
            }
        }

        private RenderRequestMultisampleLevel m_multisampleLevel = RenderRequestMultisampleLevel.x4;
        public RenderRequestMultisampleLevel MultisampleLevel
        {
            get { return m_multisampleLevel; }
            set
            {
                m_multisampleLevel = value;
                m_dirtyParams |= DirtyParams.Resolution;
            }
        }

        #endregion

        private ImageSettings m_image;
        public ImageSettings Image
        {
            get { return m_image; }
            set { m_image = value; }
        }

        public EffectSettings Effects { get; set; }
        public PostprocessingSettings Postprocessing { get; set; }
        public OverlaySettings Overlay { get; set; }



        private System.Drawing.Color m_smokeColor = System.Drawing.Color.FromArgb(242, 242, 242, 242);
        private float m_smokeTransformationSpeedCoefficient = 1f;
        private float m_smokeIntensityCoefficient = 1f;
        private float m_smokeScaleCoefficient = 1f;

        private bool m_drawNoise;
        private float m_noiseIntensityCoefficient = 1.0f;


        #endregion

        #region Helpers

        Fbo SwapBuffers()
        {
            BasicFbo tmp = m_backFbo;
            m_backFbo = m_frontFbo;
            m_frontFbo = tmp;
            return m_frontFbo;
        }

        private void SetDefaultBlending()
        {
            GL.BlendFunc(BlendingFactorSrc.One, BlendingFactorDest.OneMinusSrcAlpha);
        }

        protected virtual Matrix GetViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            if (!cameraDirection.HasValue)
                cameraDirection = Vector3.Forward;

            Matrix viewMatrix = Matrix.CreateLookAt(cameraPos, cameraPos + cameraDirection.Value, up ?? Vector3.Up);

            return viewMatrix;
        }

        #endregion

        #region Init

        public virtual void Init(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // Setup color and blending
            const int baseIntensity = 50;
            GL.ClearColor(System.Drawing.Color.FromArgb(baseIntensity, baseIntensity, baseIntensity));
            GL.BlendEquation(BlendEquationMode.FuncAdd);


            // Tileset textures and effect
            {
                // Set up tileset textures
                IEnumerable<Tileset> tilesets = world.TilesetTable.GetTilesetImages();
                TilesetImage[] tilesetImages = tilesets.Select(t =>
                    new TilesetImage(
                        t.Image.Source,
                        new Vector2I(t.Tilewidth, t.Tileheight),
                        new Vector2I(t.Spacing),
                        world.TilesetTable.TileBorder))
                    .ToArray();

                m_tilesetTexture = renderer.TextureManager.Get<TilesetTexture>(tilesetImages);


                // Set up tile grid shader
                m_effect = renderer.EffectManager.Get<NoEffectOffset>();
                renderer.EffectManager.Use(m_effect); // Need to use the effect to set uniforms

                // Set up static uniforms
                Vector2I fullTileSize = world.TilesetTable.TileSize + world.TilesetTable.TileMargins +
                                        world.TilesetTable.TileBorder * 2; // twice the border, on each side once
                Vector2 tileCount = (Vector2)m_tilesetTexture.Size / (Vector2)fullTileSize;
                m_effect.TexSizeCountUniform(new Vector3I(m_tilesetTexture.Size.X, m_tilesetTexture.Size.Y, (int)tileCount.X));
                m_effect.TileSizeMarginUniform(new Vector4I(world.TilesetTable.TileSize, world.TilesetTable.TileMargins));
                m_effect.TileBorderUniform(world.TilesetTable.TileBorder);

                m_effect.AmbientUniform(new Vector4(1, 1, 1, AmbientTerm));
            }


            // Set up geometry
            m_quad = renderer.GeometryManager.Get<FullScreenQuad>();
            m_quadOffset = renderer.GeometryManager.Get<FullScreenQuadOffset>();
        }

        protected virtual void CheckDirtyParams(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // Only setup these things when their dependency has changed (property setters enable these)

            if (m_dirtyParams.HasFlag(DirtyParams.Size))
            {
                m_gridOffset = renderer.GeometryManager.Get<FullScreenGridOffset>(GridView.Size);
                m_projMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, -1, 500);
                // Flip the image to have its origin in the top-left corner

                if (FlipYAxis)
                    m_projMatrix *= Matrix.CreateScale(1, -1, 1);

                //m_projMatrix = Matrix.CreatePerspectiveFieldOfView(MathHelper.PiOver4, 1, 1f, 500);
            }
            if (m_dirtyParams.HasFlag(DirtyParams.Resolution))
            {
                bool newRes = m_frontFbo == null || Resolution.Width != m_frontFbo.Size.X || Resolution.Height != m_frontFbo.Size.Y;

                if (newRes)
                {
                    // Reallocate front fbo
                    if (m_frontFbo != null)
                        m_frontFbo.Dispose();

                    m_frontFbo = new BasicFbo(renderer.RenderTargetManager, (Vector2I)Resolution);

                    // Reallocate back fbo; only if it was already allocated
                    if (m_backFbo != null)
                        m_backFbo.Dispose();

                    m_backFbo = new BasicFbo(renderer.RenderTargetManager, (Vector2I)Resolution);
                }

                // Reallocate MS fbo
                if (MultisampleLevel > 0)
                {
                    int multisampleCount = 1 << (int)MultisampleLevel; // 4x to 32x (4 levels)

                    if (newRes || m_fboMs == null || multisampleCount != m_fboMs.MultisampleCount)
                    {
                        if (m_fboMs != null)
                            m_fboMs.Dispose();

                        m_fboMs = new BasicFboMultisample(renderer.RenderTargetManager, (Vector2I)Resolution, multisampleCount);
                    }
                    // No need to enable Multisample capability, it is enabled automatically
                    // GL.Enable(EnableCap.Multisample);
                }
            }

            m_dirtyParams = DirtyParams.None;
        }

        #endregion

        #region Draw

        #region Callbacks

        public virtual void OnPreDraw()
        {
            var preCopyCallback = OnPreRenderingEvent;

            if (preCopyCallback != null && m_pbo != null)
                preCopyCallback(this, m_pbo.Handle);
        }

        public virtual void OnPostDraw()
        {
            var postCopyCallback = OnPostRenderingEvent;

            if (postCopyCallback != null && m_pbo != null)
                postCopyCallback(this, m_pbo.Handle);
        }

        #endregion


        public virtual void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            CheckDirtyParams(renderer, world);

            GL.Viewport(new System.Drawing.Rectangle(0, 0, Resolution.Width, Resolution.Height));

            if (MultisampleLevel > 0)
                m_fboMs.Bind();
            else
                m_frontFbo.Bind();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Enable(EnableCap.Blend);
            SetDefaultBlending();

            // View and proj transforms
            m_viewProjectionMatrix = GetViewMatrix(PositionCenterV);
            m_viewProjectionMatrix *= m_projMatrix;

            // Bind stuff to GL
            renderer.TextureManager.Bind(m_tilesetTexture);
            renderer.EffectManager.Use(m_effect);
            m_effect.TextureUniform(0);
            m_effect.DiffuseUniform(
                new Vector4(
                    1, 1, 1,
                    (1 - AmbientTerm)
                    * (Effects.EnabledEffects.HasFlag(RenderRequestEffect.DayNight)
                        ? world.Atlas.Day
                        : 1)));

            // Draw the scene
            DrawTileLayers(world);
            DrawObjectLayers(world);

            // Draw effects
            m_effectRenderer.Draw(this, renderer, world);

            // Resolve multisampling
            if (MultisampleLevel > 0)
            {
                // We have to blit to another fbo to resolve multisampling before readPixels and postprocessing, unfortunatelly
                m_fboMs.Bind(FramebufferTarget.ReadFramebuffer);
                m_frontFbo.Bind(FramebufferTarget.DrawFramebuffer);
                GL.BlitFramebuffer(
                    0, 0, m_fboMs.Size.X, m_fboMs.Size.Y,
                    0, 0, m_frontFbo.Size.X, m_frontFbo.Size.Y,
                    ClearBufferMask.ColorBufferBit, // | ClearBufferMask.DepthBufferBit, // TODO: blit depth when needed
                    BlitFramebufferFilter.Linear);
            }

            m_postprocessRenderer.Draw(this, renderer, world);
            m_overlayRenderer.Draw(this, renderer, world);

            // Copy the rendered scene
            GatherAndDistributeData(renderer);
        }

        protected virtual void DrawTileLayers(ToyWorld world)
        {
            // Set up transformation to screen space for tiles
            Matrix transform = Matrix.Identity;
            // Model transform -- scale from (-1,1) to viewSize/2, center on origin
            transform *= Matrix.CreateScale((Vector2)GridView.Size / 2);
            // World transform -- move center to view center
            transform *= Matrix.CreateTranslation(new Vector2(GridView.Center));
            // View and projection transforms
            transform *= m_viewProjectionMatrix;
            m_effect.ModelViewProjectionUniform(ref transform);


            // Draw tile layers
            List<ITileLayer> tileLayers = world.Atlas.TileLayers;
            IEnumerable<ITileLayer> toRender = tileLayers.Where(x => x.Render);
            foreach (ITileLayer tileLayer in toRender)
            {
                m_gridOffset.SetTextureOffsets(tileLayer.GetRectangle(GridView));
                m_gridOffset.Draw();
            }
        }

        protected virtual void DrawObjectLayers(ToyWorld world)
        {
            // Draw objects
            foreach (var objectLayer in world.Atlas.ObjectLayers)
            {
                // TODO: Setup for this object layer

                foreach (var gameObject in objectLayer.GetGameObjects(new RectangleF(GridView)))
                {
                    // Set up transformation to screen space for the gameObject
                    Matrix transform = Matrix.Identity;
                    // Model transform
                    IRotatable rotatableObject = gameObject as IRotatable;
                    if (rotatableObject != null)
                        transform *= Matrix.CreateRotationZ(rotatableObject.Rotation);
                    transform *= Matrix.CreateScale(gameObject.Size * 0.5f); // from (-1,1) to (-size,size)/2
                    // World transform
                    transform *= Matrix.CreateTranslation(new Vector3(gameObject.Position, 0.01f));
                    // View and projection transforms
                    transform *= m_viewProjectionMatrix;
                    m_effect.ModelViewProjectionUniform(ref transform);

                    // Setup dynamic data
                    m_quadOffset.SetTextureOffsets(gameObject.TilesetId);
                    m_quadOffset.Draw();
                }
            }
        }

        #endregion
    }
}
