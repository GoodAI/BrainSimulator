using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Textures;
using VRageMath;

namespace Render.RenderObjects.Buffers
{
    internal class BasicFboMultisample : Fbo
    {
        public int MultisampleCount { get; private set; }


        #region Genesis

        public BasicFboMultisample(RenderTargetManager renderTargetManager, Vector2I size, int multisampleCount)
        {
            MultisampleCount = multisampleCount;

            AttachTexture(FramebufferAttachment.DepthAttachment, renderTargetManager.Get<RenderTargetDepthTextureMultisample>(size, multisampleCount)); // Must be first due to error checking
            AttachTexture(FramebufferAttachment.ColorAttachment0, renderTargetManager.Get<RenderTargetColorTextureMultisample>(size, multisampleCount));

            FramebufferErrorCode err = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
            Debug.Assert(err == FramebufferErrorCode.FramebufferComplete, "Framebuffer error: " + err);
        }

        #endregion
    }
}
