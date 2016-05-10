using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Textures;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal class BasicFboMultisample : Fbo
    {
        #region Genesis

        public BasicFboMultisample(TextureManager textureManager, Vector2I size, int multisampleCount)
        {
            AttachTexture(FramebufferAttachment.DepthAttachment, textureManager.GetRenderTarget<RenderTargetDepthTextureMultisample>(size, multisampleCount)); // Must be first due to error checking
            AttachTexture(FramebufferAttachment.ColorAttachment0, textureManager.GetRenderTarget<RenderTargetColorTextureMultisample>(size, multisampleCount));

            FramebufferErrorCode err = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
            Debug.Assert(err == FramebufferErrorCode.FramebufferComplete, "Framebuffer error: " + err);
        }

        #endregion
    }
}
