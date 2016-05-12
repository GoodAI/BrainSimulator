using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Textures;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal abstract class Fbo : IDisposable
    {
        private readonly uint m_handle;

        protected readonly Dictionary<FramebufferAttachment, TextureBase> AttachedTextures =
            new Dictionary<FramebufferAttachment, TextureBase>();

        public Vector2I Size { get; private set; }


        #region Genesis

        protected Fbo()
        {
            m_handle = (uint)GL.GenFramebuffer();
        }

        public void Dispose()
        {
            GL.DeleteFramebuffer(m_handle);

            AttachedTextures.Clear();
        }

        #endregion

        #region Indexing

        protected void AttachTexture(FramebufferAttachment attachmentTarget, TextureBase texture)
        {
            Bind(FramebufferTarget.Framebuffer);

            if (texture == null)
            {
                GL.FramebufferTexture(FramebufferTarget.Framebuffer, attachmentTarget, 0, 0);
                Debug.Assert(AttachedTextures.ContainsKey(attachmentTarget));
                AttachedTextures.Remove(attachmentTarget);
                return;
            }

            Debug.Assert(
                AttachedTextures.All(pair => pair.Value.Size == texture.Size),
                "All render target sizes for a framebuffer object must be equal.");
            Size = texture.Size;

            texture.Bind();

            GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, attachmentTarget, texture.Target, texture.Handle, 0);
            AttachedTextures[attachmentTarget] = texture;
        }

        #endregion


        public void Bind(FramebufferTarget target = FramebufferTarget.Framebuffer)
        {
            GL.BindFramebuffer(target, m_handle);
        }
    }
}