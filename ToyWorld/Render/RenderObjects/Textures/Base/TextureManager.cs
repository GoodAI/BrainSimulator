using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Utils.VRageRIP.Lib.Collections;
using VRageMath;
using TexInitType = System.String;

namespace Render.RenderObjects.Textures
{
    class TilesetImage
    {
        public TilesetImage(string imagePath, Vector2I tileSize, Vector2I tileMargin, Vector2I tileBorder)
        {
            ImagePath = imagePath;
            TileSize = tileSize;
            TileMargin = tileMargin;
            TileBorder = tileBorder;
        }

        public readonly string ImagePath; // path to the tileset image (.png)
        public Vector2I TileSize; // width and height of a tile
        public Vector2I TileMargin; // number of pixels that separate one tile from another
        public Vector2I TileBorder; // pixels that should be copied and added on each side of the tile 
        // because of correct texture scaling (linear upscaling and downscaling)
    }

    internal class TextureManager
    {
        private readonly TypeSwitchParam<TextureBase, TilesetImage[]> m_textures = new TypeSwitchParam<TextureBase, TilesetImage[]>();
        private readonly TypeSwitchParam<TextureBase, Vector2I> m_renderTargetTextures = new TypeSwitchParam<TextureBase, Vector2I>();
        private readonly TypeSwitchParam<TextureBase, Vector2I, int> m_renderTargetMultisampleTextures = new TypeSwitchParam<TextureBase, Vector2I, int>();

        private readonly Dictionary<int, TextureBase> m_currentTextures = new Dictionary<int, TextureBase>();


        public TextureManager()
        {
            CaseInternal<TilesetTexture>();

            CaseRtInternal<RenderTargetColorTexture>();
            CaseRtInternal<RenderTargetDepthTexture>();

            CaseRtMsInternal<RenderTargetColorTextureMultisample>();
            CaseRtMsInternal<RenderTargetDepthTextureMultisample>();
        }

        private void CaseInternal<T>()
            where T : TextureBase
        {
            m_textures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }

        private void CaseRtInternal<T>()
            where T : TextureBase
        {
            m_renderTargetTextures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }

        private void CaseRtMsInternal<T>()
            where T : TextureBase
        {
            m_renderTargetMultisampleTextures.Case<T>((p1, p2) => (T)Activator.CreateInstance(typeof(T), p1, p2));
        }

        ////////////////////
        /// TODO: Texture caching -- we don't want to cache render target textures
        ////////////////////

        public T Get<T>(TilesetImage[] images)
            where T : TextureBase
        {
            return m_textures.Switch<T>(images);
        }

        public T GetRenderTarget<T>(Vector2I size)
            where T : TextureBase
        {
            return m_renderTargetTextures.Switch<T>(size);
        }

        public T GetRenderTarget<T>(Vector2I size, int multisampleSampleCount)
            where T : TextureBase
        {
            return m_renderTargetMultisampleTextures.Switch<T>(size, multisampleSampleCount);
        }

        public void Bind(TextureBase tex, TextureUnit texUnit = TextureUnit.Texture0)
        {
            TextureBase currTex;

            if (m_currentTextures.TryGetValue((int)texUnit, out currTex) && currTex == tex)
                return;

            GL.ActiveTexture(texUnit);
            tex.Bind();
            m_currentTextures[(int)texUnit] = tex;
        }
    }
}
