using System.Diagnostics;
using OpenTK.Graphics.OpenGL;
using RenderingBase.RenderObjects.Textures;
using VRageMath;

namespace RenderingBase.RenderObjects.Buffers
{
    public class BasicFbo : Fbo
    {
        #region Genesis

        public BasicFbo(RenderTargetManager textureManager, Vector2I size)
        {
            AttachTexture(FramebufferAttachment.DepthAttachment, textureManager.Get<RenderTargetDepthTexture>(size)); // Must be first due to error checking
            AttachTexture(FramebufferAttachment.ColorAttachment0, textureManager.Get<RenderTargetColorTexture>(size));

            FramebufferErrorCode err = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
            Debug.Assert(err == FramebufferErrorCode.FramebufferComplete, "Framebuffer error: " + err);


            // Getting of native internal texture format
            //this[FramebufferAttachment.ColorAttachment0].Bind();
            //var format = new int[20];
            //GL.GetTexLevelParameter(TextureTarget.Texture2D, 0, GetTextureParameter.TextureInternalFormat, format);
        }

        #endregion


        public TextureBase this[FramebufferAttachment index]
        {
            get { return AttachedTextures[index]; }
        }
    }
}
