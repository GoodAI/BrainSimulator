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
        public TilesetImage (string imagePath, Vector2I tileSize, Vector2I tileMargin, Vector2I tileBorder)
        {
            ImagePath = imagePath;
            TileSize = tileSize;
            TileMargin = tileMargin;
            TileBorder = tileBorder;
        }

        public string ImagePath; // path to the tileset image (.png)
        public Vector2I TileSize; // width and height of a tile
        public Vector2I TileMargin; // number of pixels that separate one tile from another
        public Vector2I TileBorder; // pixels that should be copied and added on each side of the tile 
                                    // because of correct texture scaling (linear upscaling and downscaling)
    }

    internal class TextureManager
    {
        private readonly TypeSwitchParam<TextureBase, TilesetImage[]> m_textures = new TypeSwitchParam<TextureBase, TilesetImage[]>();
        private readonly TypeSwitchParam<TextureBase, Vector2I> m_sizedTextures = new TypeSwitchParam<TextureBase, Vector2I>();

        private readonly Dictionary<int, TextureBase> m_currentTextures = new Dictionary<int, TextureBase>();


        public TextureManager()
        {
            CaseInternal<TilesetTexture>();

            CaseSizedInternal<RenderTargetColorTexture>();
            CaseSizedInternal<RenderTargetDepthTexture>();
        }

        private void CaseInternal<T>()
            where T : TextureBase
        {
            m_textures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }

        private void CaseSizedInternal<T>()
            where T : TextureBase
        {
            m_sizedTextures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }

        ////////////////////
        /// TODO: Texture caching -- je neco, co nechceme cachovat??
        ////////////////////

        public T Get<T>(TilesetImage[] images)
            where T : TextureBase
        {
            return m_textures.Switch<T>(images);
        }

        public T GetSized<T>(Vector2I size)
            where T : TextureBase
        {
            return m_sizedTextures.Switch<T>(size);
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
