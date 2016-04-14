using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using OpenTK.Graphics.OpenGL;
using Render.Tests.Textures;
using Utils.VRageRIP.Lib.Collections;

namespace Render.RenderObjects.Textures
{
    internal class TextureManager
    {
        private readonly TypeSwitch<TextureBase> m_textures = new TypeSwitch<TextureBase>();

        private readonly Dictionary<int, TextureBase> m_currentTextures = new Dictionary<int, TextureBase>();


        public TextureManager()
        {
            m_textures
                .Case<TilesetTexture>(
                    () =>
                    {
                        var str = Assembly.GetExecutingAssembly().GetManifestResourceStream("Render.Tests.Textures." + "roguelike_selection_summer.png");
                        Debug.Assert(str != null);
                        return new TilesetTexture(str);
                    });

            // TODO: use:
            //CaseInternal<TilesetTexture>();
        }

        private void CaseInternal<T>()
            where T : TextureBase, new()
        {
            m_textures.Case<T>(() => new T());
        }


        public T Get<T>()
            where T : TextureBase
        {
            return m_textures.Switch<T>();
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
