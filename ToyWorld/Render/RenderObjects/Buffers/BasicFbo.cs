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
            // TODO: Redo when caching in manager is completed
            this[FramebufferAttachment.DepthAttachment] = textureManager.GetSized<RenderTargetDepthTexture>(size); // Must be first due to error checking
            this[FramebufferAttachment.ColorAttachment0] = textureManager.GetSized<RenderTargetColorTexture>(size);

            FramebufferErrorCode err = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
            Debug.Assert(err == FramebufferErrorCode.FramebufferComplete, "Framebuffer error: " + err);
        }
    
        #endregion
    }
}
