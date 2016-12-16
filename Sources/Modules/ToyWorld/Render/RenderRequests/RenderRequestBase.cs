using System;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using RenderingBase.Renderer;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using RenderingBase.RenderRequests;
using VRageMath;
using World.ToyWorldCore;
using Rectangle = VRageMath.Rectangle;
using RectangleF = VRageMath.RectangleF;
using TupleType = System.Tuple<World.Atlas.Layers.ITileLayer, int[], VRageMath.Vector4I[]>;

namespace Render.RenderRequests
{
    public abstract class RenderRequestBase
        : IRenderRequestBaseInternal<ToyWorld>
    {
        [Flags]
        internal enum DirtyParam
        {
            None = 0,
            Size = 1,
        }

        internal enum TextureBindPosition
        {
            SummerTileset = 0,
            WinterTileset = 1,

            TileTypes = 4,

            Ui = 6,

            PostEffectTextureBindPosition = 8,
        }


        #region Fields

        internal GameObjectPainter GameObjectPainter;
        internal EffectPainter EffectPainter;
        internal PostprocessPainter PostprocessPainter;
        internal OverlayPainter OverlayPainter;
        internal ImagePainter ImagePainter;

        protected internal BasicFbo FrontFbo, BackFbo;
        protected internal BasicFboMultisample FboMs;

        protected internal Quad Quad;

        protected internal Matrix ProjMatrix;
        protected internal Matrix ViewProjectionMatrix;

        internal DirtyParam DirtyParams;

        #endregion

        #region Genesis

        protected RenderRequestBase()
        {
            GameObjectPainter = new GameObjectPainter(this);
            EffectPainter = new EffectPainter(this);
            PostprocessPainter = new PostprocessPainter(this);
            OverlayPainter = new OverlayPainter(this);
            ImagePainter = new ImagePainter(this);

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

            Quad.Dispose();

            GameObjectPainter.Dispose();
            EffectPainter.Dispose();
            PostprocessPainter.Dispose();
            OverlayPainter.Dispose();
            ImagePainter.Dispose();
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

        protected float PositionZ { get { return PositionCenterV.Z; } set { PositionCenterV = new Vector3(PositionCenterV2, value); } }

        private Vector2 m_sizeV;
        protected Vector2 SizeV
        {
            get { return m_sizeV; }
            set
            {
                const float minSize = 0.01f;
                const float maxSize = 50; // Texture max size is (1 << 14) / 8, (8 is max layer count), sqrt of that is max size of one side
                m_sizeV = new Vector2(Math.Min(maxSize, Math.Max(minSize, value.X)), Math.Min(maxSize, Math.Max(minSize, value.Y)));
                DirtyParams |= DirtyParam.Size;
            }
        }

        protected internal virtual RectangleF ViewV { get { return new RectangleF(Vector2.Zero, SizeV) { Center = new Vector2(PositionCenterV) }; } }

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
                const int minResolution = 2;
                const int maxResolution = 4096;
                if (value.Width < minResolution || value.Height < minResolution)
                    throw new ArgumentOutOfRangeException("value", "Invalid resolution: must be greater than " + minResolution + "x" + minResolution+" pixels.");
                if (value.Width > maxResolution || value.Height > maxResolution)
                    throw new ArgumentOutOfRangeException("value", "Invalid resolution: must be at most " + maxResolution + "x" + maxResolution + " pixels.");

                m_resolution = value;
            }
        }

        public RenderRequestMultisampleLevel MultisampleLevel { get; set; }

        #endregion

        #region Settings

        public GameObjectSettings GameObjects { get; set; }
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

        protected virtual Matrix Get2DViewMatrix(Vector3 cameraPos, Vector3? up = null)
        {
            return Matrix.CreateLookAt(cameraPos, cameraPos + Vector3.Forward, up ?? Vector3.Up);
        }

        protected virtual Matrix Get3DViewMatrix(Vector3 cameraPos, Vector3? cameraDirection = null, Vector3? up = null)
        {
            cameraDirection = cameraDirection ?? Vector3.Forward;
            up = up ?? Vector3.Backward;

            Vector3 cross = Vector3.Cross(cameraDirection.Value, up.Value); // Perpendicular to both
            cross = Vector3.Cross(cross, cameraDirection.Value); // Up vector closest to the original up

            return Matrix.CreateLookAt(cameraPos, cameraPos + cameraDirection.Value, cross);
        }

        internal TextureUnit GetTextureUnit(TextureBindPosition bindPosition)
        {
            return TextureUnit.Texture0 + (int)bindPosition;
        }

        #endregion

        #region Init

        public virtual void Init()
        {
            PositionCenterV = new Vector3(PositionCenterV2, !GameObjects.Use3D ? 2 : 2.4f);

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

            // Setup geometry
            Quad = Renderer.GeometryManager.Get<Quad>();


            GameObjectPainter.Init(Renderer, World, GameObjects);
            EffectPainter.Init(Renderer, World, Effects);
            PostprocessPainter.Init(Renderer, World, Postprocessing);
            OverlayPainter.Init(Renderer, World, Overlay);
            ImagePainter.Init(Renderer, World, Image);
        }

        protected virtual void CheckDirtyParams()
        {
            // Update renderers
            GameObjectPainter.CheckDirtyParams(Renderer, World);


            // Only setup these things when their dependency has changed (property setters enable these)

            if (DirtyParams.HasFlag(DirtyParam.Size))
            {
                if (!GameObjects.Use3D)
                    ProjMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, -1, 100);
                else
                    ProjMatrix = Matrix.CreatePerspectiveFieldOfView(MathHelper.PiOver4, 1, 0.1f, 500);

                // Flip the image to have its origin in the top-left corner
                if (FlipYAxis)
                    ProjMatrix *= Matrix.CreateScale(1, -1, 1);
            }


            DirtyParams = DirtyParam.None;
        }

        #endregion

        #region Draw

        public virtual void Update()
        {
            CheckDirtyParams();

            // View and proj transforms
            ViewProjectionMatrix = !GameObjects.Use3D ? Get2DViewMatrix(PositionCenterV) : Get3DViewMatrix(PositionCenterV);
            ViewProjectionMatrix *= ProjMatrix;
        }


        #region Events

        public virtual void OnPreDraw()
        {
            if (GameObjectPainter != null)
                GameObjectPainter.OnPreDraw();

            if (ImagePainter != null)
                ImagePainter.OnPreDraw();
        }

        public virtual void OnPostDraw()
        {
            // Copy the rendered scene -- doing this here lets GL time to finish the scene
            if (ImagePainter != null)
                ImagePainter.Draw(Renderer, World);

            if (GameObjectPainter != null)
                GameObjectPainter.OnPostDraw();

            if (ImagePainter != null)
                ImagePainter.OnPostDraw();
        }

        #endregion


        public virtual void Draw()
        {
            GL.Viewport(new System.Drawing.Rectangle(0, 0, Resolution.Width, Resolution.Height));

            if (MultisampleLevel > 0)
                FboMs.Bind();
            else
                FrontFbo.Bind();

            // Setup stuff
            GL.DepthMask(true); // enables writing to depth buffer - needed for GL.Clear

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            // Draw the scene
            GameObjectPainter.Draw(Renderer, World);

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
                if (ImagePainter.Settings.CopyDepth || EffectPainter.Settings.EnabledEffects != RenderRequestEffect.None)
                    GL.BlitFramebuffer(
                        0, 0, FboMs.Size.X, FboMs.Size.Y,
                        0, 0, FrontFbo.Size.X, FrontFbo.Size.Y,
                        ClearBufferMask.DepthBufferBit,
                        BlitFramebufferFilter.Nearest);
            }

            // Draw effects after multisampling to save fragment shader calls
            EffectPainter.Draw(Renderer, World);

            // Depth testing is useless for these
            GL.Disable(EnableCap.DepthTest);

            PostprocessPainter.Draw(Renderer, World);
            OverlayPainter.Draw(Renderer, World);

            // Tell OpenGL driver to submit any unissued commands to the GPU
            GL.Flush();
        }

        #endregion
    }
}
