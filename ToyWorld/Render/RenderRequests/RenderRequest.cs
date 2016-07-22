using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
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
using TupleType = System.Tuple<World.Atlas.Layers.ITileLayer, int[], VRageMath.Vector4I[]>;

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

        protected internal BasicTexture1D TileTypesTexure;
        private Pbo<ushort> m_tileTypesBuffer;
        internal readonly ushort[] LocalTileTypesBuffer = new ushort[1];

        protected internal DuplicatedCubeGrid Grid;
        protected internal Cube Cube;
        protected internal Quad Quad;

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

            PositionCenterV = new Vector3(0, 0, 5);
            SizeV = new Vector2(3, 3);
            Resolution = new System.Drawing.Size(1024, 1024);

            MultisampleLevel = RenderRequestMultisampleLevel.x4;
        }

        public virtual void Dispose()
        {
            UnregisterRenderRequest();

            if (FrontFbo != null)
                FrontFbo.Dispose();
            if (BackFbo != null)
                BackFbo.Dispose();
            if (FboMs != null)
                FboMs.Dispose();

            Effect.Dispose();

            TilesetTexture.Dispose();
            TileTypesTexure.Dispose();
            m_tileTypesBuffer.Dispose();

            if (Grid != null) // It is initialized during Draw
                Grid.Dispose();
            Cube.Dispose();
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

        public void UnregisterRenderRequest()
        {
            Renderer.RemoveRenderRequest(this);
        }

        public RendererBase<ToyWorld> Renderer { get; set; }
        public ToyWorld World { get; set; }


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

        #region Settings

        public EffectSettings Effects { get; set; }
        public PostprocessingSettings Postprocessing { get; set; }
        public OverlaySettings Overlay { get; set; }
        public ImageSettings Image { get; set; }

        #endregion

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
            //cameraDirection = Vector3.Forward;
            cameraDirection = cameraDirection ?? new Vector3(0, 5f, -1);
            //up = up ?? Vector3.Up;
            up = up ?? Vector3.Backward;

            Vector3 cross = Vector3.Cross(cameraDirection.Value, up.Value); // Perpendicular to both
            cross = Vector3.Cross(cross, cameraDirection.Value); // Up vector closest to the original up

            Matrix viewMatrix = Matrix.CreateLookAt(cameraPos - cameraDirection.Value * 10, cameraPos, cross);

            return viewMatrix;
        }

        #endregion

        #region Init

        public virtual void Init()
        {
            // Set up color and blending
            const int baseIntensity = 50;
            GL.ClearColor(System.Drawing.Color.FromArgb(baseIntensity, baseIntensity, baseIntensity));
            GL.BlendEquation(BlendEquationMode.FuncAdd);
            GL.DepthFunc(DepthFunction.Less); // Default value?
            //GL.DepthFunc(DepthFunction.Always); // Ignores stored depth values, but still writes them

            // Set up framebuffers
            {
                bool newRes = FrontFbo == null || Resolution.Width != FrontFbo.Size.X || Resolution.Height != FrontFbo.Size.Y;

                if (newRes)
                {
                    // Reallocate front fbo
                    if (FrontFbo != null)
                        FrontFbo.Dispose();

                    FrontFbo = new BasicFbo(Renderer.RenderTargetManager, (Vector2I)Resolution);

                    // Reallocate back fbo; only if it was already allocated
                    if (BackFbo != null)
                        BackFbo.Dispose();

                    BackFbo = new BasicFbo(Renderer.RenderTargetManager, (Vector2I)Resolution);
                }

                // Reallocate MS fbo
                if (MultisampleLevel > 0)
                {
                    int multisampleCount = 1 << (int)MultisampleLevel; // 4x to 32x (4 levels)

                    if (newRes || FboMs == null || multisampleCount != FboMs.MultisampleCount)
                    {
                        if (FboMs != null)
                            FboMs.Dispose();

                        FboMs = new BasicFboMultisample(Renderer.RenderTargetManager, (Vector2I)Resolution, multisampleCount);
                    }
                    // No need to enable Multisample capability, it is enabled automatically
                    // GL.Enable(EnableCap.Multisample);
                }
            }

            // Tileset textures
            {
                // Set up tileset textures
                IEnumerable<Tileset> tilesets = World.TilesetTable.GetTilesetImages();
                TilesetImage[] tilesetImages = tilesets.Select(t =>
                        new TilesetImage(
                            t.Image.Source,
                            new Vector2I(t.Tilewidth, t.Tileheight),
                            new Vector2I(t.Spacing),
                            World.TilesetTable.TileBorder))
                    .ToArray();

                TilesetTexture = Renderer.TextureManager.Get<TilesetTexture>(tilesetImages);
            }

            // Set up tile grid shader
            {
                Effect = Renderer.EffectManager.Get<NoEffectOffset>();
                Renderer.EffectManager.Use(Effect); // Need to use the effect to set uniforms

                // Set up static uniforms
                Vector2I fullTileSize = World.TilesetTable.TileSize + World.TilesetTable.TileMargins +
                                        World.TilesetTable.TileBorder * 2; // twice the border, on each side once
                Vector2 tileCount = (Vector2)TilesetTexture.Size / (Vector2)fullTileSize;
                Effect.TexSizeCountUniform(new Vector3I(TilesetTexture.Size.X, TilesetTexture.Size.Y, (int)tileCount.X));
                Effect.TileSizeMarginUniform(new Vector4I(World.TilesetTable.TileSize, World.TilesetTable.TileMargins));
                Effect.TileBorderUniform(World.TilesetTable.TileBorder);

                Effect.AmbientUniform(new Vector4(1, 1, 1, EffectRenderer.AmbientTerm));
            }

            // Set up geometry
            Quad = Renderer.GeometryManager.Get<Quad>();
            Cube = Renderer.GeometryManager.Get<Cube>();

            EffectRenderer.Init(Renderer, World, Effects);
            PostprocessRenderer.Init(Renderer, World, Postprocessing);
            OverlayRenderer.Init(Renderer, World, Overlay);
            ImageRenderer.Init(Renderer, World, Image);
        }

        protected virtual void CheckDirtyParams()
        {
            // Only setup these things when their dependency has changed (property setters enable these)

            if (DirtyParams.HasFlag(DirtyParam.Size))
            {
                Grid = Renderer.GeometryManager.Get<DuplicatedCubeGrid>(GridView.Size);

                ProjMatrix = Matrix.CreatePerspectiveFieldOfView(MathHelper.PiOver4, 1, 1f, 500);
                //ProjMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, -1, 10);

                // Flip the image to have its origin in the top-left corner
                if (FlipYAxis)
                    ProjMatrix *= Matrix.CreateScale(1, -1, 1);
            }

            DirtyParams = DirtyParam.None;
        }

        #endregion

        #region Draw

        protected IEnumerable<ITileLayer> GetTileLayersToRender()
        {
            return World.Atlas.TileLayers.Where(l => l.Render);
        }

        #region Events

        public virtual void OnPreDraw()
        {
            // Start asynchronous copying of tile types
            Rectangle gridView = GridView;
            int tileCount = gridView.Size.Size();

            ITileLayer[] toRender = GetTileLayersToRender().ToArray();

            // Reallocate stuff if needed -- texture holds tileTypes for all the layers
            int totalTileCount = tileCount * toRender.Length;
            Debug.Assert(totalTileCount < 2 << 13, "TileTypesTexture will overflow!");

            if (TileTypesTexure == null || totalTileCount > TileTypesTexure.Size.Size())
            {
                // Init buffer
                if (m_tileTypesBuffer != null)
                    m_tileTypesBuffer.Dispose();
                m_tileTypesBuffer = new Pbo<ushort>(1);
                m_tileTypesBuffer.Init(totalTileCount, hint: BufferUsageHint.StreamDraw);

                // Init texture
                if (TileTypesTexure != null)
                    TileTypesTexure.Dispose();
                TileTypesTexure = Renderer.TextureManager.Get<BasicTexture1D>(new Vector2I(totalTileCount, 1));
                TileTypesTexure.DefaultInit();
            }

            // Start data copying
            for (int i = 0; i < toRender.Length; i++)
            {
                // Store data directly to device memory
                m_tileTypesBuffer.Bind();
                IntPtr bufferPtr = GL.MapBuffer(m_tileTypesBuffer.Target, BufferAccess.WriteOnly);
                toRender[i].GetTileTypesAt(gridView, bufferPtr, tileCount, i * tileCount);
                GL.UnmapBuffer(m_tileTypesBuffer.Target);

                // Start async copying to the texture
                m_tileTypesBuffer.Bind(BufferTarget.PixelUnpackBuffer);
                TileTypesTexure.Update1D(tileCount, dataType: PixelType.UnsignedShort, offset: i * tileCount, byteDataOffset: i * tileCount * sizeof(ushort));
            }


            if (ImageRenderer != null)
                ImageRenderer.OnPreDraw();
        }

        public virtual void OnPostDraw()
        {
            // Copy the rendered scene -- doing this here lets GL time to finish the scene
            if (ImageRenderer != null)
                ImageRenderer.Draw(Renderer, World);


            if (ImageRenderer != null)
                ImageRenderer.OnPostDraw();
        }

        #endregion


        public virtual void Update()
        {
            CheckDirtyParams();

            // View and proj transforms
            ViewProjectionMatrix = GetViewMatrix(PositionCenterV);
            ViewProjectionMatrix *= ProjMatrix;
        }

        public virtual void Draw()
        {
            GL.Viewport(new System.Drawing.Rectangle(0, 0, Resolution.Width, Resolution.Height));

            if (MultisampleLevel > 0)
                FboMs.Bind();
            else
                FrontFbo.Bind();

            // Setup stuff
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            //GL.Enable(EnableCap.Blend);
            GL.Disable(EnableCap.Blend);
            SetDefaultBlending();
            GL.Enable(EnableCap.DepthTest);

            // Bind stuff to GL
            Renderer.TextureManager.Bind(TilesetTexture[0]);
            Renderer.TextureManager.Bind(TilesetTexture[1], TextureUnit.Texture1);
            Renderer.TextureManager.Bind(TileTypesTexure, TextureUnit.Texture4);
            Renderer.EffectManager.Use(Effect);
            Effect.TextureUniform(0);
            Effect.TextureWinterUniform(1);
            Effect.TileTypesTextureUniform(4);
            Effect.DiffuseUniform(new Vector4(1, 1, 1, EffectRenderer.GetGlobalDiffuseComponent(World)));
            Effect.TileVertexCountUniform(Grid.FaceCount * 4);


            // Draw the scene
            DrawTileLayers();
            DrawObjectLayers();

            // Resolve multisampling
            if (MultisampleLevel > 0)
            {
                // We have to blit to another fbo to resolve multisampling before readPixels and postprocessing, unfortunatelly
                FboMs.Bind(FramebufferTarget.ReadFramebuffer);
                FrontFbo.Bind(FramebufferTarget.DrawFramebuffer);
                GL.BlitFramebuffer(
                    0, 0, FboMs.Size.X, FboMs.Size.Y,
                    0, 0, FrontFbo.Size.X, FrontFbo.Size.Y,
                    ClearBufferMask.ColorBufferBit,
                    BlitFramebufferFilter.Linear);
                if (ImageRenderer.Settings.CopyDepth)
                    GL.BlitFramebuffer(
                        0, 0, FboMs.Size.X, FboMs.Size.Y,
                        0, 0, FrontFbo.Size.X, FrontFbo.Size.Y,
                        ClearBufferMask.DepthBufferBit,
                        BlitFramebufferFilter.Nearest);
            }

            // Effects should not be used with depth testing
            GL.Disable(EnableCap.DepthTest);

            // Draw effects after multisampling to save fragment shader calls
            EffectRenderer.Draw(Renderer, World);

            PostprocessRenderer.Draw(Renderer, World);
            OverlayRenderer.Draw(Renderer, World);

            // Tell OpenGL driver to submit any unissued commands to the GPU
            GL.Flush();
        }

        protected virtual void DrawTileLayers()
        {
            Rectangle gridView = GridView;
            int tileCount = gridView.Size.Size();

            // Set up transformation to screen space for tiles
            Matrix transform = Matrix.Identity;
            // Model transform -- scale from (-1,1) to viewSize/2, center on origin
            transform *= Matrix.CreateScale((Vector2)gridView.Size / 2);

            // Draw tile layers
            int i = 0;

            foreach (var tileLayer in GetTileLayersToRender())
            {
                // World transform -- move center to view center
                Matrix t = transform * Matrix.CreateScale(1, 1, tileLayer.Thickness / 2);
                t *= Matrix.CreateTranslation(new Vector3(gridView.Center, tileLayer.SpanIntervalFrom + tileLayer.Thickness / 2));
                // View and projection transforms
                t *= ViewProjectionMatrix;
                Effect.ModelViewProjectionUniform(ref t);
                Effect.TileTypesIdxOffsetUniform(i++ * tileCount);

                // Using the tileTypes texture should block until the data is fully copied from the pbos (onPreDraw)
                Grid.Draw();
            }
        }

        protected virtual void DrawObjectLayers()
        {
            Effect.TileTypesIdxOffsetUniform(0);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);

            // Draw objects
            foreach (var objectLayer in World.Atlas.ObjectLayers)
            {
                Matrix layerTransform = Matrix.CreateTranslation(0, 0, objectLayer.SpanIntervalFrom);

                foreach (var gameObject in objectLayer.GetGameObjects(new RectangleF(GridView)))
                {
                    // Set up transformation to screen space for the gameObject
                    Matrix transform = Matrix.Identity;
                    // Model transform
                    IRotatable rotatableObject = gameObject as IRotatable;
                    if (rotatableObject != null)
                        transform *= Matrix.CreateRotationZ(rotatableObject.Rotation);
                    transform *= Matrix.CreateScale(new Vector3(gameObject.Size, objectLayer.Thickness) * 0.5f); // from (-1,1) to (-size,size)/2
                    // World transform
                    transform *= Matrix.CreateTranslation(gameObject.Position);
                    transform *= layerTransform;
                    // View and projection transforms
                    transform *= ViewProjectionMatrix;
                    Effect.ModelViewProjectionUniform(ref transform);

                    // Setup dynamic data
                    LocalTileTypesBuffer[0] = (ushort)gameObject.TilesetId;
                    TileTypesTexure.Update1D(1, dataType: PixelType.UnsignedShort, data: LocalTileTypesBuffer);
                    Cube.Draw();
                }
            }
        }

        #endregion
    }
}
