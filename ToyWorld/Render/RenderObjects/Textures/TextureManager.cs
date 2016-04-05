using OpenTK.Graphics.OpenGL;
using Render.Tests.Textures;
using Utils.VRageRIP.Lib.Collections;

namespace Render.RenderObjects.Textures
{
    internal class TextureManager
    {
        private readonly TypeSwitch<TextureBase> m_textures = new TypeSwitch<TextureBase>();


        public TextureManager()
        {




            // TODO: jak dostat cesty k obrazkum z asi World?!?
            // TODO: load uvs do VAO




            m_textures
                .Case<TilesetTexture>(() =>
                    new TilesetTexture(null));
        }


        public void BindTexture<T>(TextureUnit texUnit = TextureUnit.Texture0)
            where T : TextureBase
        {
            // TODO: pamatovat unit--texture binding, nedelat ho znova

            GL.ActiveTexture(texUnit);

            var tex = m_textures.Switch<T>();
            tex.Bind();
        }
    }
}
