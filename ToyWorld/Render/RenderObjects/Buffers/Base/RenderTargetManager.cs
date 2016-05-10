using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Textures;
using Utils.VRageRIP.Lib.Collections;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal class RenderTargetManager
    {
        private readonly TypeSwitchParam<TextureBase, Vector2I> m_renderTargetTextures = new TypeSwitchParam<TextureBase, Vector2I>();
        private readonly TypeSwitchParam<TextureBase, Vector2I, int> m_renderTargetMultisampleTextures = new TypeSwitchParam<TextureBase, Vector2I, int>();

        private readonly Dictionary<int, TextureBase> m_currentTextures = new Dictionary<int, TextureBase>();


        public RenderTargetManager()
        {
            CaseInternal<RenderTargetColorTexture>();
            CaseInternal<RenderTargetDepthTexture>();

            CaseMsInternal<RenderTargetColorTextureMultisample>();
            CaseMsInternal<RenderTargetDepthTextureMultisample>();
        }

        private void CaseInternal<T>()
            where T : TextureBase
        {
            m_renderTargetTextures.Case<T>(i => (T)Activator.CreateInstance(typeof(T), i));
        }

        private void CaseMsInternal<T>()
            where T : TextureBase
        {
            m_renderTargetMultisampleTextures.Case<T>((p1, p2) => (T)Activator.CreateInstance(typeof(T), p1, p2));
        }


        public T Get<T>(Vector2I size)
            where T : TextureBase
        {
            return m_renderTargetTextures.Switch<T>(size);
        }

        public T Get<T>(Vector2I size, int multisampleSampleCount)
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
