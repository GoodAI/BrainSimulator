using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Textures;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal class Fbo : IDisposable
    {
        private readonly uint m_handle;

        private readonly Dictionary<FramebufferAttachment, TextureBase> m_attachedTextures =
            new Dictionary<FramebufferAttachment, TextureBase>();



        public Vector2I Size { get; private set; }


        #region Genesis

        public Fbo()
        {
            m_handle = (uint) GL.GenFramebuffer();
        }

        public void Dispose()
        {
            GL.DeleteFramebuffer(m_handle);

            m_attachedTextures.Clear();
        }

        #endregion

        #region Indexing

        protected TextureBase this[FramebufferAttachment attachmentTarget]
        {
            get { return m_attachedTextures[attachmentTarget]; }
            set { AttachTexture(attachmentTarget, value); }
        }

        private void AttachTexture(FramebufferAttachment attachmentTarget, TextureBase texture)
        {
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, m_handle);

            if (texture == null)
            {
                GL.FramebufferTexture(FramebufferTarget.Framebuffer, attachmentTarget, 0, 0);
                Debug.Assert(m_attachedTextures.ContainsKey(attachmentTarget));
                m_attachedTextures.Remove(attachmentTarget);
                return;
            }

            Debug.Assert(
                m_attachedTextures.All(pair => pair.Value.Size == texture.Size), 
                "All render target sizes for a framebuffer object must be equal.");
            Size = texture.Size;

            texture.Bind();

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, attachmentTarget, texture.Target, texture.Handle, 0);
            m_attachedTextures[attachmentTarget] = texture;

            //GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
        }

        #endregion


        public void Bind(FramebufferTarget target = FramebufferTarget.Framebuffer)
        {
            GL.BindFramebuffer(target, m_handle);
        }
    }
}