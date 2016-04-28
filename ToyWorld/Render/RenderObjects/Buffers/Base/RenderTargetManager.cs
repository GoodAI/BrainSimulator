using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Textures;
using Utils.VRageRIP.Lib.Collections;
using TexInitType = System.String;

namespace Render.RenderObjects.Buffers
{
    internal class RenderTargetManager
    {
        private readonly TypeSwitchParam<TextureBase, TilesetImage[]> m_textures = new TypeSwitchParam<TextureBase, TilesetImage[]>();

        private readonly Dictionary<int, TextureBase> m_currentTextures = new Dictionary<int, TextureBase>();


        public RenderTargetManager()
        {
            CaseInternal<TilesetTexture>();
            //m_textures
            //    .Case<TilesetTexture>(
            //        () =>
            //        {
            //            var str = Assembly.GetExecutingAssembly().GetManifestResourceStream("Render.Tests.Textures." + "roguelike_selection_summer.png");
            //            Debug.Assert(str != null);
            //            return new TilesetTexture(str);
            //        });
        }

        private void CaseInternal<T>()
            where T : TextureBase
        {
            m_textures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }


        public T Get<T>(TilesetImage[] tilesetImages)
            where T : TextureBase
        {
            return m_textures.Switch<T>(tilesetImages);
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
