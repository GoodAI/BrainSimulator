using System;
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


        #region View control properties

        /// <summary>
        /// The position of the center of view.
        /// </summary>
        protected Vector3 PositionCenterV { get; set; }
        /// <summary>
        /// The position of the center of view. Equivalent to PositionCenterV (except for the z value).
        /// </summary>
        protected Vector2 PositionCenterV2 { get { return new Vector2(PositionCenterV); } set { PositionCenterV = new Vector3(value, PositionCenterV.Z); } }
        protected Vector2 SizeV { get; set; }
        protected RectangleF ViewV { get { return new RectangleF(Vector2.Zero, SizeV) { Center = new Vector2(PositionCenterV) }; } }

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

        #endregion

        #region Genesis

        public RenderRequest()
        {
            PositionCenterV = new Vector3(0, 0, 20);
            SizeV = new Vector2(3, 3);
            Resolution = new System.Drawing.Size(1024, 1024);
            Image = new uint[0];
        }

        public virtual void Dispose()
        {
            m_effect.Dispose();
            m_tex.Dispose();
            m_grid.Dispose();
            m_quad.Dispose();
        }

        #endregion

        #region IRenderRequestBase overrides

        public System.Drawing.PointF PositionCenter
        {
            get { return new System.Drawing.PointF(PositionCenterV.X, PositionCenterV.Y); }
            protected set { PositionCenterV2 = new Vector2(value.X, value.Y); }
        }

        public System.Drawing.SizeF Size
        {
            get { return new System.Drawing.SizeF(SizeV.X, SizeV.Y); }
            set
            {
                if (value.Width <= 0 || value.Height <= 0)
                    throw new ArgumentOutOfRangeException("value", "Cannot use non-positive view size for a render request. Value: " + value);

                SizeV = new Vector2(value.Width, value.Height);
            }
        }

        public System.Drawing.RectangleF View { get { return new System.Drawing.RectangleF(PositionCenter, Size); } }


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
                    throw new ArgumentOutOfRangeException("value", "Invalid resolution: must be smaller than " + maxResolution + " pixels.");

                m_resolution = value;
            }
        }


        public bool GatherImage { get; set; }
        public uint[] Image { get; private set; }

        #endregion


        protected Matrix GetViewMatrix(Vector3 rotation, Vector3 cameraPos, Vector3 cameraDirection = default(Vector3))
        {
            var viewMatrix = Matrix.Identity;

            if (rotation.X > 0)
                viewMatrix = Matrix.CreateRotationZ(rotation.X);

            if (rotation.Y > 0)
                viewMatrix *= Matrix.CreateRotationX(rotation.Y);

            if (rotation.Z > 0)
                viewMatrix *= Matrix.CreateRotationY(rotation.Z);

            if (cameraDirection == default(Vector3))
                cameraDirection = Vector3.Forward;
            viewMatrix *= Matrix.CreateLookAt(cameraPos, cameraPos + cameraDirection, Vector3.Up);

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
            m_projMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, -1, 500);
            //m_projMatrix = Matrix.CreatePerspectiveFieldOfView(MathHelper.PiOver4, 1, 1f, 500);
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
            m_viewProjectionMatrix = GetViewMatrix(Vector3.Zero, PositionCenterV);
            m_viewProjectionMatrix *= m_projMatrix;
            m_effect.SetUniformMatrix4(m_mvpPos, transform * m_viewProjectionMatrix);


            // Draw tile layers
            foreach (var tileLayer in world.Atlas.TileLayers)
            {
                //transform *= Matrix.CreateTranslation(0, 0, -0.1f);
                //m_effect.SetUniformMatrix4(m_mvpPos, transform * m_viewProjectionMatrix);

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


            // Gather data to host mem
            if (GatherImage)
            {
                if (Image.Length < Resolution.Width * Resolution.Height)
                    Image = new uint[Resolution.Width * Resolution.Height];

                GL.ReadPixels(0, 0, Resolution.Width, Resolution.Height, PixelFormat.Bgra, PixelType.UnsignedByte, Image);
            }
        }
    }
}
