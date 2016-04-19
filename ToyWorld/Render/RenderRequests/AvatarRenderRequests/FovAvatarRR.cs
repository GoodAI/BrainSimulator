using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.Renderer;
using Render.RenderObjects.Geometries;
using Render.Tests.Effects;
using Render.Tests.Textures;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests.AvatarRenderRequests
{
    internal class FovAvatarRR : AvatarRRBase, IFovAvatarRR
    {
        private int[] m_buffer;

        private NoEffectOffset m_effect;
        private TilesetTexture m_tex;
        private FullScreenGrid m_grid;
        private FullScreenQuadOffset m_quad;

        private Matrix m_projMatrix;
        private Matrix m_worldMatrix;
        private Matrix m_worldViewProjectionMatrix;
        private int m_mvpPos;


        #region Genesis

        internal FovAvatarRR(int avatarID)
            : base(avatarID)
        { }

        public override void Dispose()
        {
            m_effect.Dispose();
            m_tex.Dispose();
            m_grid.Dispose();
            m_quad.Dispose();
            base.Dispose();
        }

        #endregion

        #region IFovAvatarRR overrides

        public uint[] Image { get; private set; }

        #endregion

        #region RenderRequestBase overrides

        public override void Init(RendererBase renderer, ToyWorld world)
        {
            Image = new uint[renderer.Width * renderer.Height];

            GL.ClearColor(System.Drawing.Color.DimGray);
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
            Vector2I fullTileSize = world.TilesetTable.TileSize + world.TilesetTable.TileMargins;
            Vector2 tileCount = (Vector2)m_tex.Size / (Vector2)fullTileSize;
            m_effect.SetUniform3(m_effect.GetUniformLocation("texSizeCount"), new Vector3I(m_tex.Size.X, m_tex.Size.Y, (int)tileCount.X));
            m_effect.SetUniform4(m_effect.GetUniformLocation("tileSizeMargin"), new Vector4I(world.TilesetTable.TileSize, world.TilesetTable.TileMargins));
            m_mvpPos = m_effect.GetUniformLocation("mvp");

            SizeV = world.Size;
            PositionCenterV = (Vector2)SizeV * 0.5f;
            ViewV = new RectangleF(Vector2.Zero, (Vector2)SizeV);

            // Set up tile grid geometry
            m_buffer = new int[SizeV.Size()];
            m_grid = renderer.GeometryManager.Get<FullScreenGrid>(SizeV);
            m_quad = renderer.GeometryManager.Get<FullScreenQuadOffset>();

            // Scale up to world size (was (-1,1) originally)
            m_worldMatrix = Matrix.CreateScale(world.Size.X * 0.5f, world.Size.Y * 0.5f, 1);
            // No view matrix needed here -- we are fixed on origin (center of the world),
            // or the view has to be computed each step
            m_projMatrix = Matrix.CreateOrthographic(SizeV.X, SizeV.Y, 1, -20);
        }

        public override void Draw(RendererBase renderer, ToyWorld world)
        {
            GL.Clear(ClearBufferMask.ColorBufferBit);

            renderer.EffectManager.Use(m_effect);
            renderer.TextureManager.Bind(m_tex);


            // Set up transformation to screen space
            m_worldViewProjectionMatrix = m_worldMatrix;

            // No view transform
            //if (Rotation.X > 0)
            //m_worldViewProjectionMatrix *= Matrix.CreateRotationZ(Rotation.X);

            m_worldViewProjectionMatrix *= m_projMatrix;
            m_effect.SetUniformMatrix4(m_mvpPos, m_worldViewProjectionMatrix);


            // Draw tile layers
            foreach (var tileLayer in world.Atlas.TileLayers)
            {
                tileLayer.GetRectangle(ViewV.Position, ViewV.Size, m_buffer);
                m_grid.SetTextureOffsets(m_buffer);

                m_grid.Draw();
            }


            // Draw objects
            foreach (var objectLayer in world.Atlas.ObjectLayers)
            {
                // TODO: Setup for this object layer
                foreach (var gameObject in objectLayer.GetGameObjects(ViewV))
                {
                    Matrix modelMatrix = Matrix.Identity;
                    modelMatrix *= Matrix.CreateRotationZ(0.5f);
                    modelMatrix *= Matrix.CreateScale(0.2f);
                    modelMatrix *= Matrix.CreateTranslation(0.3f, 0, 0.01f);
                    modelMatrix *= m_worldViewProjectionMatrix;
                    m_effect.SetUniformMatrix4(m_mvpPos, modelMatrix);

                    const int tileSetOffset = 17;
                    m_quad.SetTextureOffsets(tileSetOffset);

                    m_quad.Draw();
                }
            }

            GL.ReadPixels(0, 0, renderer.Width, renderer.Height, PixelFormat.Bgra, PixelType.UnsignedByte, Image);
        }

        #endregion
    }
}
