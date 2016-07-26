using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Effects;
using RenderingBase.Renderer;
using RenderingBase.RenderObjects.Buffers;
using RenderingBase.RenderObjects.Geometries;
using RenderingBase.RenderObjects.Textures;
using TmxMapSerializer.Elements;
using VRageMath;
using World.Atlas.Layers;
using World.GameActors.GameObjects;
using World.Physics;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class GameObjectRenderer
        : RRRendererBase<GameObjectSettings, RenderRequestBase>, IDisposable
    {
        #region Fields

        protected internal NoEffectOffset Effect;

        protected internal TilesetTexture TilesetTexture;

        protected internal BasicTexture1D TileTypesTexure;
        private Pbo<ushort> m_tileTypesBuffer;
        internal readonly ushort[] LocalTileTypesBuffer = new ushort[15];

        protected internal GeometryBase Grid;
        protected internal GeometryBase Cube;


        private ITileLayer[] m_toRender;
        private Rectangle m_gridView;
        public HashSet<IGameObject> IgnoredGameObjects = new HashSet<IGameObject>();

        #endregion

        #region Genesis

        public GameObjectRenderer(RenderRequestBase owner)
            : base(owner)
        { }

        public virtual void Dispose()
        {
            Effect.Dispose();

            TilesetTexture.Dispose();
            TileTypesTexure.Dispose();
            m_tileTypesBuffer.Dispose();

            if (Grid != null) // It is initialized during Draw
                Grid.Dispose();
            if (Cube != null)
                Cube.Dispose();
        }

        #endregion

        #region Init

        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world, GameObjectSettings settings)
        {
            Settings = settings;

            GL.DepthFunc(DepthFunction.Lequal);
            //GL.DepthFunc(DepthFunction.Always); // Ignores stored depth values, but still writes them

            // Tileset textures
            {
                // Set up tileset textures
                IEnumerable<Tileset> tilesets = world.TilesetTable.GetTilesetImages();
                TilesetImage[] tilesetImages = tilesets.Select(
                    t =>
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
            if (settings.Use3D)
                Cube = renderer.GeometryManager.Get<DuplicatedCube>();
            else
                Cube = renderer.GeometryManager.Get<Quad>();

            IgnoredGameObjects.Clear();
        }

        public void CheckDirtyParams(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            m_gridView = Owner.GridView;


            // Get currently rendered layers
            m_toRender = GetTileLayersToRender().ToArray();

            if (Owner.DirtyParams.HasFlag(RenderRequestBase.DirtyParam.Size))
            {
                if (Settings.Use3D)
                    Grid = renderer.GeometryManager.Get<DuplicatedCubeGrid>(m_gridView.Size);
                else
                    Grid = renderer.GeometryManager.Get<DuplicatedGrid>(m_gridView.Size);

                // Reallocate stuff if needed -- texture holds tileTypes for all the layers
                int totalTileCount = m_gridView.Size.Size() * m_toRender.Length;
                Debug.Assert(totalTileCount < 1 << 14, "TileTypesTexture will overflow!");

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
                    TileTypesTexure = renderer.TextureManager.Get<BasicTexture1D>(new Vector2I(totalTileCount, 1));
                    TileTypesTexure.DefaultInit();
                }
            }
        }

        #endregion

        #region Draw

        protected virtual IEnumerable<ITileLayer> GetTileLayersToRender()
        {
            return Owner.World.Atlas.TileLayers.Where(l => l.Render);
        }

        #region Callbacks

        public virtual void OnPreDraw()
        {
            if (Settings.EnabledGameObjects == RenderRequestGameObject.None)
                return;

            m_gridView = Owner.GridView; // The value might have changed

            if (Settings.EnabledGameObjects.HasFlag(RenderRequestGameObject.TileLayers))
            {
                // Start asynchronous copying of tile types
                int tileCount = m_gridView.Size.Size();

                for (int i = 0; i < m_toRender.Length; i++)
                {
                    // Store data directly to device memory
                    m_tileTypesBuffer.Bind();
                    IntPtr bufferPtr = GL.MapBuffer(m_tileTypesBuffer.Target, BufferAccess.WriteOnly);
                    m_toRender[i].GetTileTypesAt(m_gridView, bufferPtr, tileCount, i * tileCount);
                    GL.UnmapBuffer(m_tileTypesBuffer.Target);

                    // Start async copying to the texture
                    m_tileTypesBuffer.Bind(BufferTarget.PixelUnpackBuffer);
                    TileTypesTexure.Update1D(tileCount, dataType: PixelType.UnsignedShort, offset: i * tileCount, byteDataOffset: i * tileCount * sizeof(ushort));
                }
            }
        }

        public virtual void OnPostDraw()
        { }

        #endregion

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            if (Settings.EnabledGameObjects == RenderRequestGameObject.None)
                return;

            m_gridView = Owner.GridView; // The value might have changed

            if (!Settings.Use3D)
            {
                GL.Enable(EnableCap.Blend);
                Owner.SetDefaultBlending();
            }
            else
                GL.Disable(EnableCap.Blend);

            GL.Enable(EnableCap.DepthTest);
            GL.DepthMask(true);

            // Bind stuff to GL
            renderer.TextureManager.Bind(TilesetTexture[0], Owner.GetTextureUnit(RenderRequestBase.TextureBindPosition.SummerTileset));
            renderer.TextureManager.Bind(TilesetTexture[1], Owner.GetTextureUnit(RenderRequestBase.TextureBindPosition.WinterTileset));
            renderer.TextureManager.Bind(TileTypesTexure, Owner.GetTextureUnit(RenderRequestBase.TextureBindPosition.TileTypes));
            renderer.EffectManager.Use(Effect);
            Effect.TextureUniform((int)RenderRequestBase.TextureBindPosition.SummerTileset);
            Effect.TextureWinterUniform((int)RenderRequestBase.TextureBindPosition.WinterTileset);
            Effect.TileTypesTextureUniform((int)RenderRequestBase.TextureBindPosition.TileTypes);
            Effect.DiffuseUniform(new Vector4(1, 1, 1, Owner.EffectRenderer.GetGlobalDiffuseComponent(world)));
            Effect.TileVertexCountUniform(Settings.Use3D ? DuplicatedCubeGrid.FaceCount * 4 : 4);

            if (Settings.EnabledGameObjects.HasFlag(RenderRequestGameObject.TileLayers))
                DrawTileLayers(renderer, world);
            if (Settings.EnabledGameObjects.HasFlag(RenderRequestGameObject.ObjectLayers))
                DrawObjectLayers(renderer, world);
        }

        protected virtual void DrawTileLayers(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            int tileCount = m_gridView.Size.Size();

            // Draw tile layers
            int i = 0;

            foreach (var tileLayer in GetTileLayersToRender())
            {
                // Set up transformation to screen space for tiles
                // Model transform -- scale from (-1,1) to viewSize/2, center on origin
                Matrix transform = Matrix.CreateScale(new Vector3(m_gridView.Size, tileLayer.Thickness) * 0.5f);
                // World transform -- move center to view center
                transform *= Matrix.CreateTranslation(new Vector3(m_gridView.Center, tileLayer.SpanIntervalFrom));// + tileLayer.Thickness / 2));
                // View and projection transforms
                transform *= Owner.ViewProjectionMatrix;
                Effect.ModelViewProjectionUniform(ref transform);
                Effect.TileTypesIdxOffsetUniform(i++ * tileCount);

                // Using the tileTypes texture should block until the data is fully copied from the pbos (onPreDraw)
                Grid.Draw();
            }
        }

        protected virtual void DrawObjectLayers(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            Effect.TileTypesIdxOffsetUniform(0);
            GL.BindBuffer(BufferTarget.PixelUnpackBuffer, 0);

            // Draw objects
            foreach (var objectLayer in world.Atlas.ObjectLayers)
            {
                foreach (var gameObject in objectLayer.GetGameObjects(new RectangleF(m_gridView)))
                {
                    if (IgnoredGameObjects.Contains(gameObject))
                        continue;

                    // Set up transformation to screen space for the gameObject
                    Matrix transform = Matrix.Identity;
                    // Model transform
                    IRotatable rotatableObject = gameObject as IRotatable;
                    if (rotatableObject != null)
                        transform *= Matrix.CreateRotationZ(rotatableObject.Rotation);
                    transform *= Matrix.CreateScale(new Vector3(gameObject.Size, objectLayer.Thickness) * 0.5f); // from (-1,1) to (-size,size)/2
                    // World transform
                    transform *= Matrix.CreateTranslation(new Vector3(gameObject.Position, objectLayer.SpanIntervalFrom + objectLayer.Thickness / 2));
                    // View and projection transforms
                    transform *= Owner.ViewProjectionMatrix;
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
