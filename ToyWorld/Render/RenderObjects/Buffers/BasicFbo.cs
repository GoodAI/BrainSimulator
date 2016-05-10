using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Textures;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal class BasicFbo : Fbo
    {
        #region Genesis

        public BasicFbo(TextureManager textureManager, Vector2I size)
        {
            AttachTexture(FramebufferAttachment.DepthAttachment, textureManager.GetRenderTarget<RenderTargetDepthTexture>(size)); // Must be first due to error checking
            AttachTexture(FramebufferAttachment.ColorAttachment0, textureManager.GetRenderTarget<RenderTargetColorTexture>(size));

            FramebufferErrorCode err = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
            Debug.Assert(err == FramebufferErrorCode.FramebufferComplete, "Framebuffer error: " + err);
        }

        #endregion
    }
}
