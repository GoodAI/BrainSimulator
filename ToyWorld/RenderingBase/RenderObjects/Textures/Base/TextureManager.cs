using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Utils.VRageRIP.Lib.Collections;
using VRageMath;

namespace RenderingBase.RenderObjects.Textures
{
    public class TextureManager
    {
        private readonly TypeSwitchParam<TextureBase, TilesetImage[]> m_tileTextures = new TypeSwitchParam<TextureBase, TilesetImage[]>();
        private readonly TypeSwitchParam<TextureBase, Vector2I> m_textures = new TypeSwitchParam<TextureBase, Vector2I>();
        private readonly Dictionary<int, TextureBase> m_currentTextures = new Dictionary<int, TextureBase>();


        public TextureManager()
        {
            CaseInternal<BasicTexture>();
            TileCaseInternal<TilesetTexture>();
        }

        private void TileCaseInternal<T>()
            where T : TextureBase
        {
            m_tileTextures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }

        private void CaseInternal<T>()
            where T : TextureBase
        {
            m_textures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }


        ////////////////////
        /// TODO: Texture caching -- we don't want to cache render target textures
        ////////////////////

        public T Get<T>(params TilesetImage[] images)
            where T : TextureBase
        {
            return m_tileTextures.Switch<T>(images);
        }

        public T Get<T>(Vector2I dimensions)
            where T : TextureBase
        {
            return m_textures.Switch<T>(dimensions);
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
