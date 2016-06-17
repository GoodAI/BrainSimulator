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
        protected internal enum DirtyParam
        {
            None = 0,
            Size = 1,
        }


        #region Fields

        internal EffectRenderer EffectRenderer;
        internal PostprocessRenderer PostprocessRenderer;
        internal OverlayRenderer OverlayRenderer;
        internal ImageRenderer ImageRenderer;

        protected internal BasicFbo FrontFbo, BackFbo;
        protected internal BasicFboMultisample FboMs;

        protected internal NoEffectOffset Effect;

        protected internal TilesetTexture TilesetTexture;

        protected internal FullScreenGridOffset GridOffset;
        protected internal FullScreenQuadOffset QuadOffset;
        protected internal FullScreenQuad Quad;

        protected internal Matrix ProjMatrix;
        protected internal Matrix ViewProjectionMatrix;

        protected internal DirtyParam DirtyParams;

        #endregion

        #region Genesis

        protected RenderRequest()
        {
            EffectRenderer = new EffectRenderer(this);
            PostprocessRenderer = new PostprocessRenderer(this);
            OverlayRenderer = new OverlayRenderer(this);
            ImageRenderer = new ImageRenderer(this);

            PositionCenterV = new Vector3(0, 0, 20);
            SizeV = new Vector2(3, 3);
            Resolution = new System.Drawing.Size(1024, 1024);

            MultisampleLevel = RenderRequestMultisampleLevel.x4;
        }

        public virtual void Dispose()
        {
            if (FrontFbo != null)
                FrontFbo.Dispose();
            if (BackFbo != null)
                BackFbo.Dispose();
            if (FboMs != null)
                FboMs.Dispose();

            Effect.Dispose();

            TilesetTexture.Dispose();

            if (GridOffset != null) // It is initialized during Draw
                GridOffset.Dispose();
            QuadOffset.Dispose();
            Quad.Dispose();

            EffectRenderer.Dispose();
            PostprocessRenderer.Dispose();
            OverlayRenderer.Dispose();
            ImageRenderer.Dispose();
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
                DirtyParams |= DirtyParam.Size;
            }
        }

        protected internal virtual RectangleF ViewV { get { return new RectangleF(Vector2.Zero, SizeV) { Center = new Vector2(PositionCenterV) }; } }

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
                DirtyParams |= DirtyParam.Size;
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
            }
        }

        public RenderRequestMultisampleLevel MultisampleLevel { get; set; }

        #endregion


        public EffectSettings Effects { get; set; }
        public PostprocessingSettings Postprocessing { get; set; }
        public OverlaySettings Overlay { get; set; }
        public ImageSettings Image { get; set; }


        #endregion

        #region Helpers

        internal Fbo SwapBuffers()
        {
            BasicFbo tmp = BackFbo;
            BackFbo = FrontFbo;
            FrontFbo = tmp;
            return FrontFbo;
        }

        internal void SetDefaultBlending()
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
            // Set up color and blending
            const int baseIntensity = 50;
            GL.ClearColor(System.Drawing.Color.FromArgb(baseIntensity, baseIntensity, baseIntensity));
            GL.BlendEquation(BlendEquationMode.FuncAdd);

            // Set up framebuffers
            {
                bool newRes = FrontFbo == null || Resolution.Width != FrontFbo.Size.X || Resolution.Height != FrontFbo.Size.Y;

                if (newRes)
                {
                    // Reallocate front fbo
                    if (FrontFbo != null)
                        FrontFbo.Dispose();

                    FrontFbo = new BasicFbo(renderer.RenderTargetManager, (Vector2I)Resolution);

                    // Reallocate back fbo; only if it was already allocated
                    if (BackFbo != null)
                        BackFbo.Dispose();

                    BackFbo = new BasicFbo(renderer.RenderTargetManager, (Vector2I)Resolution);
                }

                // Reallocate MS fbo
                if (MultisampleLevel > 0)
                {
                    int multisampleCount = 1 << (int)MultisampleLevel; // 4x to 32x (4 levels)

                    if (newRes || FboMs == null || multisampleCount != FboMs.MultisampleCount)
                    {
                        if (FboMs != null)
                            FboMs.Dispose();

                        FboMs = new BasicFboMultisample(renderer.RenderTargetManager, (Vector2I)Resolution, multisampleCount);
                    }
                    // No need to enable Multisample capability, it is enabled automatically
                    // GL.Enable(EnableCap.Multisample);
                }
            }

            // Tileset textures
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

                TilesetTexture = renderer.TextureManager.Get<TilesetTexture>(tilesetImages);
            }

            // Set up tile grid shader
            {
                Effect = renderer.EffectManager.Get<NoEffectOffset>();
                renderer.EffectManager.Use(Effect); // Need to use the effect to set uniforms

                // Set up static uniforms
                Vector2I fullTileSize = world.TilesetTable.TileSize + world.TilesetTable.TileMargins +
                                        world.TilesetTable.TileBorder * 2; // twice the border, on each side once
                Vector2 tileCount = (Vector2)TilesetTexture.Size / (Vector2)fullTileSize;
                Effect.TexSizeCountUniform(new Vector3I(TilesetTexture.Size.X, TilesetTexture.Size.Y, (int)tileCount.X));
                Effect.TileSizeMarginUniform(new Vector4I(world.TilesetTable.TileSize, world.TilesetTable.TileMargins));
                Effect.TileBorderUniform(world.TilesetTable.TileBorder);

                Effect.AmbientUniform(new Vector4(1, 1, 1, EffectRenderer.AmbientTerm));
            }

            // Set up geometry
            Quad = renderer.GeometryManager.Get<FullScreenQuad>();
            QuadOffset = renderer.GeometryManager.Get<FullScreenQuadOffset>();

            // Initialize renderers
            EffectRenderer.Init(renderer, world, Effects);
            PostprocessRenderer.Init(renderer, world, Postprocessing);
            OverlayRenderer.Init(renderer, world, Overlay);
            ImageRenderer.Init(renderer, world, Image);
        }

        protected virtual void CheckDirtyParams(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // Only setup these things when their dependency has changed (property setters enable these)

            if (DirtyParams.HasFlag(DirtyParam.Size))
            {
                GridOffset = renderer.GeometryManager.Get<FullScreenGridOffset>(GridView.Size);
                ProjMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, -1, 500);
                // Flip the image to have its origin in the top-left corner

                if (FlipYAxis)
                    ProjMatrix *= Matrix.CreateScale(1, -1, 1);

                //m_projMatrix = Matrix.CreatePerspectiveFieldOfView(MathHelper.PiOver4, 1, 1f, 500);
            }

            DirtyParams = DirtyParam.None;
        }

        #endregion

        #region Draw

        #region Callbacks

        public virtual void OnPreDraw()
        {
            if (ImageRenderer != null)
                ImageRenderer.OnPreDraw();
        }

        public virtual void OnPostDraw()
        {
            if (ImageRenderer != null)
                ImageRenderer.OnPostDraw();
        }

        #endregion


        public virtual void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            CheckDirtyParams(renderer, world);

            GL.Viewport(new System.Drawing.Rectangle(0, 0, Resolution.Width, Resolution.Height));

            if (MultisampleLevel > 0)
                FboMs.Bind();
            else
                FrontFbo.Bind();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Enable(EnableCap.Blend);
            SetDefaultBlending();

            // View and proj transforms
            ViewProjectionMatrix = GetViewMatrix(PositionCenterV);
            ViewProjectionMatrix *= ProjMatrix;

            // Bind stuff to GL
            renderer.TextureManager.Bind(TilesetTexture);
            renderer.EffectManager.Use(Effect);
            Effect.TextureUniform(0);
            Effect.DiffuseUniform(new Vector4(1, 1, 1, EffectRenderer.GetGlobalDiffuseComponent(world)));

            // Draw the scene
            DrawTileLayers(world);
            DrawObjectLayers(world);

            // Draw effects
            EffectRenderer.Draw(renderer, world);

            // Resolve multisampling
            if (MultisampleLevel > 0)
            {
                // We have to blit to another fbo to resolve multisampling before readPixels and postprocessing, unfortunatelly
                FboMs.Bind(FramebufferTarget.ReadFramebuffer);
                FrontFbo.Bind(FramebufferTarget.DrawFramebuffer);
                GL.BlitFramebuffer(
                    0, 0, FboMs.Size.X, FboMs.Size.Y,
                    0, 0, FrontFbo.Size.X, FrontFbo.Size.Y,
                    ClearBufferMask.ColorBufferBit, // | ClearBufferMask.DepthBufferBit, // TODO: blit depth when needed
                    BlitFramebufferFilter.Linear);
            }

            PostprocessRenderer.Draw(renderer, world);
            OverlayRenderer.Draw(renderer, world);

            // Copy the rendered scene
            ImageRenderer.Draw(renderer, world);
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
            transform *= ViewProjectionMatrix;
            Effect.ModelViewProjectionUniform(ref transform);


            // Draw tile layers
            List<ITileLayer> tileLayers = world.Atlas.TileLayers;
            IEnumerable<ITileLayer> toRender = tileLayers.Where(x => x.Render);

            foreach (ITileLayer tileLayer in toRender)
            {
                GridOffset.SetTextureOffsets(tileLayer.GetRectangle(GridView));
                GridOffset.Draw();
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
                    transform *= ViewProjectionMatrix;
                    Effect.ModelViewProjectionUniform(ref transform);

                    // Setup dynamic data
                    QuadOffset.SetTextureOffsets(gameObject.TilesetId);
                    QuadOffset.Draw();
                }
            }
        }

        #endregion
    }
}
