using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Utils.VRageRIP.Lib.Collections;

namespace RenderingBase.RenderObjects.Textures
{
    public class TextureManager
    {
        private readonly TypeSwitchParam<TextureBase, TilesetImage[]> m_textures = new TypeSwitchParam<TextureBase, TilesetImage[]>();
        private readonly Dictionary<int, TextureBase> m_currentTextures = new Dictionary<int, TextureBase>();


        public TextureManager()
        {
            CaseInternal<TilesetTexture>();

        }

        private void CaseInternal<T>()
            where T : TextureBase
        {
            m_textures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }


        ////////////////////
        /// TODO: Texture caching -- we don't want to cache render target textures
        ////////////////////

        public T Get<T>(TilesetImage[] images)
            where T : TextureBase
        {
            return m_textures.Switch<T>(images);
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
