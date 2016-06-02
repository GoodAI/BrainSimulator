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
using VRageMath;
using World.Atlas.Layers;
using World.Physics;
using World.ToyWorldCore;
using FullScreenQuadOffset = Render.RenderObjects.Geometries.FullScreenQuadOffset;
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
        }


        #region Fields

        const TextureUnit PostEffectTextureBindPosition = TextureUnit.Texture6;
        const float AmbientTerm = 0.25f;

        public bool CopyToWindow { get; set; }

        private BasicFbo m_frontFbo, m_backFbo;
        private BasicFboMultisample m_fboMs;

        private NoEffectOffset m_effect;
        private SmokeEffect m_smokeEffect;
        private NoiseEffect m_noiseEffect;

        private TilesetTexture m_tex;

        private FullScreenGridTex m_grid;
        private FullScreenQuadOffset m_quadOffset;
        private FullScreenQuad m_quad;

        private Pbo m_pbo;

        private Matrix m_projMatrix;
        private Matrix m_viewProjectionMatrix;

        protected DirtyParams m_dirtyParams;

        #endregion

        #region Genesis

        protected RenderRequest()
        {
            PositionCenterV = new Vector3(0, 0, 20);
            SizeV = new Vector2(3, 3);
            Resolution = new System.Drawing.Size(1024, 1024);
            Image = new uint[0];
            CopyImageThroughCpu = false;
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
            if (m_smokeEffect != null)
                m_smokeEffect.Dispose();
            if (m_noiseEffect != null)
                m_noiseEffect.Dispose();

            m_tex.Dispose();

            if (m_grid != null) // It is initialized during Draw
                m_grid.Dispose();
            m_quadOffset.Dispose();
            m_quad.Dispose();

            if (m_pbo != null)
                m_pbo.Dispose();
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

        private int m_multisampleLevel = 2;
        public int MultisampleLevel
        {
            get { return m_multisampleLevel; }
            set
            {
                const int minSamples = 0;
                const int maxSamples = 4;
                if (value < minSamples)
                    throw new ArgumentOutOfRangeException("value", "Invalid multisample level: must be positive.");
                if (value > maxSamples)
                    throw new ArgumentOutOfRangeException("value", "Invalid multisample level: must be at most " + maxSamples + ".");

                m_multisampleLevel = value;
                m_dirtyParams |= DirtyParams.Resolution;
            }
        }

        #endregion

        #region Image

        private bool m_gatherImage;
        public bool GatherImage
        {
            get { return m_gatherImage; }
            set
            {
                m_gatherImage = value;
                m_dirtyParams |= DirtyParams.Image;
            }
        }

        private bool m_copyImageThroughCpu;
        public bool CopyImageThroughCpu
        {
            get { return m_copyImageThroughCpu; }
            set
            {
                m_copyImageThroughCpu = value;
                m_dirtyParams |= DirtyParams.Image;
            }
        }

        public uint[] Image { get; private set; }

        public event Action<IRenderRequestBase, uint> OnPreRenderingEvent;
        public event Action<IRenderRequestBase, uint> OnPostRenderingEvent;

        #endregion

        #region Effects - overlay

        private bool m_drawSmoke;
        private System.Drawing.Color m_smokeColor = System.Drawing.Color.FromArgb(242, 242, 242, 242);
        private float m_smokeTransformationSpeedCoefficient = 1f;
        private float m_smokeIntensityCoefficient = 1f;
        private float m_smokeScaleCoefficient = 1f;

        public bool DrawSmoke
        {
            get { return m_drawSmoke; }
            set
            {
                m_drawSmoke = value;
                m_dirtyParams |= DirtyParams.Smoke;
            }
        }
        public System.Drawing.Color SmokeColor
        {
            get { return m_smokeColor; }
            set
            {
                m_smokeColor = value;
                m_dirtyParams |= DirtyParams.Smoke;
            }
        }
        public float SmokeTransformationSpeedCoefficient
        {
            get { return m_smokeTransformationSpeedCoefficient; }
            set { m_smokeTransformationSpeedCoefficient = value; }
        }
        public float SmokeIntensityCoefficient
        {
            get { return m_smokeIntensityCoefficient; }
            set { m_smokeIntensityCoefficient = value; }
        }
        public float SmokeScaleCoefficient
        {
            get { return m_smokeScaleCoefficient; }
            set { m_smokeScaleCoefficient = value; }
        }

        #endregion

        #region Effects - post

        protected bool PostProcessingActive { get { return DrawNoise; } }

        private bool m_drawNoise;
        private float m_noiseIntensityCoefficient = 1.0f;

        public bool DrawNoise
        {
            get { return m_drawNoise; }
            set
            {
                m_drawNoise = value;
                m_dirtyParams |= DirtyParams.Noise;
            }
        }
        public float NoiseIntensityCoefficient
        {
            get { return m_noiseIntensityCoefficient; }
            set { m_noiseIntensityCoefficient = value; }
        }

        #endregion

        #endregion

        #region Helpers

        Fbo SwapBuffers()
        {
            BasicFbo tmp = m_backFbo;
            m_backFbo = m_frontFbo;
            m_frontFbo = tmp;
            return m_frontFbo;
        }

        #endregion

        #region Init

        public virtual void Init(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // Setup color and blending
            const int baseIntensity = 50;
            GL.ClearColor(System.Drawing.Color.FromArgb(baseIntensity, baseIntensity, baseIntensity));
            GL.BlendEquation(BlendEquationMode.FuncAdd);
            GL.BlendFunc(BlendingFactorSrc.One, BlendingFactorDest.OneMinusSrcAlpha);


            // Set up tileset textures
            string[] tilesetImagePaths = world.TilesetTable.GetTilesetImages();
            TilesetImage[] tilesetImages = new TilesetImage[tilesetImagePaths.Length];

            for (int i = 0; i < tilesetImages.Length; i++)
                tilesetImages[i] = new TilesetImage(tilesetImagePaths[i], world.TilesetTable.TileSize,
                                                    world.TilesetTable.TileMargins, world.TilesetTable.TileBorder);

            m_tex = renderer.TextureManager.Get<TilesetTexture>(tilesetImages);


            // Set up tile grid shader
            m_effect = renderer.EffectManager.Get<NoEffectOffset>();
            renderer.EffectManager.Use(m_effect); // Need to use the effect to set uniforms
            m_effect.TextureUniform(0);

            // Set up static uniforms
            Vector2I fullTileSize = world.TilesetTable.TileSize + world.TilesetTable.TileMargins +
                world.TilesetTable.TileBorder * 2; // twice the border, on each side once
            Vector2 tileCount = (Vector2)m_tex.Size / (Vector2)fullTileSize;
            m_effect.TexSizeCountUniform(new Vector3I(m_tex.Size.X, m_tex.Size.Y, (int)tileCount.X));
            m_effect.TileSizeMarginUniform(new Vector4I(world.TilesetTable.TileSize, world.TilesetTable.TileMargins));
            m_effect.TileBorderUniform(world.TilesetTable.TileBorder);

            m_effect.AmbientUniform(new Vector4(255, 255, 255, AmbientTerm));


            // Set up geometry
            m_quad = renderer.GeometryManager.Get<FullScreenQuad>();
            m_quadOffset = renderer.GeometryManager.Get<FullScreenQuadOffset>();


            // Set up pixel buffer object for data transfer to RR issuer; don't allocate any memory (it's done in CheckDirtyParams)
            if (!CopyImageThroughCpu)
                m_pbo = new Pbo();


            // Don't call CheckDirtyParams here because stuff like Resolution can be set by the user only after Init is called.
        }

        private void CheckDirtyParams(RendererBase<ToyWorld> renderer)
        {
            // Only setup these things when their dependency has changed (property setters enable these)

            if (m_dirtyParams.HasFlag(DirtyParams.Size))
            {
                m_grid = renderer.GeometryManager.Get<FullScreenGridTex>(GridView.Size);
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

                    if (DrawNoise)
                        m_dirtyParams |= DirtyParams.Noise; // Force Noise re-checking (we need to set viewportSize uniform)
                }

                // Reallocate MS fbo
                if (MultisampleLevel > 0)
                {
                    int multisampleCount = 1 << MultisampleLevel; // 4x to 32x (4 levels)

                    if (MultisampleLevel == 1)
                        multisampleCount = 4; // 2x does not seem to be working, force it to be at least 4x

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
            if (m_dirtyParams.HasFlag(DirtyParams.Image))
            {
                if (CopyImageThroughCpu)
                {
                    if (!GatherImage)
                        Image = new uint[0];
                    else if (Image.Length < Resolution.Width * Resolution.Height)
                        Image = new uint[Resolution.Width * Resolution.Height];
                }
                else
                {
                    if (m_pbo.ByteCount != Resolution.Width * Resolution.Height * sizeof(uint))
                        m_pbo.Init(Resolution.Width * Resolution.Height, null, BufferUsageHint.StreamDraw);
                }
            }
            if (m_dirtyParams.HasFlag(DirtyParams.Smoke))
            {
                if (m_smokeEffect == null)
                    m_smokeEffect = renderer.EffectManager.Get<SmokeEffect>();
                renderer.EffectManager.Use(m_smokeEffect); // Need to use the effect to set uniforms
                m_smokeEffect.SmokeColorUniform(new Vector4(SmokeColor.R, SmokeColor.G, SmokeColor.B, SmokeColor.A) / 255f);
            }
            if (m_dirtyParams.HasFlag(DirtyParams.Noise))
            {
                if (m_noiseEffect == null)
                    m_noiseEffect = renderer.EffectManager.Get<NoiseEffect>();
                renderer.EffectManager.Use(m_noiseEffect); // Need to use the effect to set uniforms
                m_noiseEffect.ViewportSizeUniform((Vector2I)Resolution);
                m_noiseEffect.SceneTextureUniform((int)PostEffectTextureBindPosition - (int)TextureUnit.Texture0);
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
            CheckDirtyParams(renderer);

            GL.Viewport(new System.Drawing.Rectangle(0, 0, Resolution.Width, Resolution.Height));

            if (MultisampleLevel > 0)
                m_fboMs.Bind();
            else
                m_frontFbo.Bind();

            GL.Clear(ClearBufferMask.ColorBufferBit);
            GL.Enable(EnableCap.Blend);

            // View and proj transforms
            m_viewProjectionMatrix = GetViewMatrix(PositionCenterV);
            m_viewProjectionMatrix *= m_projMatrix;

            // Bind stuff to GL
            renderer.TextureManager.Bind(m_tex);
            renderer.EffectManager.Use(m_effect);
            m_effect.DiffuseUniform(new Vector4(255, 255, 255, (1 - AmbientTerm) * world.Atlas.Day));

            // Draw the scene
            DrawTileLayers(world);
            DrawObjectLayers(world);

            // Draw effects
            DrawEffects(renderer);

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

            // Apply post-processing
            if (PostProcessingActive) // If any postprocessing effect is active
            {
                GL.Disable(EnableCap.Blend);
                ApplyPostProcessingEffects(renderer);
            }

            // Copy the rendered scene
            GatherAndDistributeData(renderer);
        }

        protected virtual Matrix GetViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            if (!cameraDirection.HasValue)
                cameraDirection = Vector3.Forward;

            Matrix viewMatrix = Matrix.CreateLookAt(cameraPos, cameraPos + cameraDirection.Value, up ?? Vector3.Up);

            return viewMatrix;
        }

        private void DrawTileLayers(ToyWorld world)
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
                m_grid.SetTextureOffsets(tileLayer.GetRectangle(GridView));
                m_grid.Draw();
            }
        }

        private void DrawObjectLayers(ToyWorld world)
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

        private void DrawEffects(RendererBase<ToyWorld> renderer)
        {
            if (DrawSmoke)
            {
                renderer.EffectManager.Use(m_smokeEffect);

                // Set up transformation to world and screen space for noise effect
                Matrix transform = Matrix.Identity;
                // Model transform -- scale from (-1,1) to viewSize/2, center on origin
                transform *= Matrix.CreateScale(ViewV.Size / 2);
                // World transform -- move center to view center
                transform *= Matrix.CreateTranslation(new Vector3(ViewV.Center, 1f));
                m_smokeEffect.ModelWorldUniform(ref transform);
                // View and projection transforms
                transform *= m_viewProjectionMatrix;
                m_smokeEffect.ModelViewProjectionUniform(ref transform);

                // Advance noise time by a visually pleasing step; wrap around if we run for waaaaay too long.
                double step = 0.005d * SmokeTransformationSpeedCoefficient;
                double seed = renderer.SimTime * step % 3e6d;
                m_smokeEffect.TimeStepUniform(new Vector2((float)seed, (float)step));
                m_smokeEffect.MeanScaleUniform(new Vector2(SmokeIntensityCoefficient, SmokeScaleCoefficient));

                m_quad.Draw();
            }

            // more stufffs
        }

        private void ApplyPostProcessingEffects(RendererBase<ToyWorld> renderer)
        {
            // Always draw post-processing from the front to the back buffer
            m_backFbo.Bind();

            if (DrawNoise)
            {
                renderer.EffectManager.Use(m_noiseEffect);
                renderer.TextureManager.Bind(m_frontFbo[FramebufferAttachment.ColorAttachment0], PostEffectTextureBindPosition); // Use data from front Fbo

                // Advance noise time by a visually pleasing step; wrap around if we run for waaaaay too long.
                double step = 0.005d;
                double seed = renderer.SimTime * step % 3e6d;
                m_noiseEffect.TimeStepUniform(new Vector2((float)seed, (float)step));
                m_noiseEffect.VarianceUniform(NoiseIntensityCoefficient);

                m_quad.Draw();
            }

            SwapBuffers();

            // more stuffs

            // The final scene should be left in the front buffer
        }

        private void GatherAndDistributeData(RendererBase<ToyWorld> renderer)
        {
            if (CopyToWindow)
            {
                // TODO: TEMP: copy to default framebuffer (our window) -- will be removed
                m_frontFbo.Bind(FramebufferTarget.ReadFramebuffer);
                GL.BindFramebuffer(FramebufferTarget.DrawFramebuffer, 0);
                GL.BlitFramebuffer(
                    0, 0, m_frontFbo.Size.X, m_frontFbo.Size.Y,
                    0, 0, renderer.Width, renderer.Height,
                    ClearBufferMask.ColorBufferBit,
                    BlitFramebufferFilter.Nearest);
            }

            // Gather data to host mem
            if (GatherImage)
            {
                m_frontFbo.Bind();
                GL.ReadBuffer(ReadBufferMode.ColorAttachment0); // Works for fbo bound to Framebuffer (not DrawFramebuffer)

                if (CopyImageThroughCpu)
                {
                    GL.BindBuffer(BufferTarget.PixelPackBuffer, 0);
                    GL.ReadPixels(0, 0, Resolution.Width, Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, Image);
                }
                else
                {
                    m_pbo.Bind();
                    GL.ReadPixels(0, 0, Resolution.Width, Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, default(IntPtr));
                }
            }
        }

        #endregion
    }
}
