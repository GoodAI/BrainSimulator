using System;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Effects;
using RenderingBase.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class PostprocessRenderer
        : RRRendererBase<PostprocessingSettings>, IDisposable
    {
        #region Fields

        protected const TextureUnit PostEffectTextureBindPosition = TextureUnit.Texture6;

        protected NoiseEffect m_noiseEffect;

        #endregion

        #region Genesis

        public virtual void Dispose()
        {
            if (m_noiseEffect != null)
                m_noiseEffect.Dispose();
        }

        #endregion

        #region Init

        public virtual void Init(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world, PostprocessingSettings settings)
        {
            if (m_noiseEffect == null)
                m_noiseEffect = renderer.EffectManager.Get<NoiseEffect>();
            renderer.EffectManager.Use(m_noiseEffect); // Need to use the effect to set uniforms
            m_noiseEffect.ViewportSizeUniform((Vector2I)Resolution);
            m_noiseEffect.SceneTextureUniform((int)PostEffectTextureBindPosition - (int)TextureUnit.Texture0);
        }

        #endregion

        #region Draw

        public virtual void Draw(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // Always draw post-processing from the front to the back buffer
            m_backFbo.Bind();

            if (DrawNoise)
            {
                renderer.EffectManager.Use(m_noiseEffect);
                renderer.TextureManager.Bind(m_frontFbo[FramebufferAttachment.ColorAttachment0], PostEffectTextureBindPosition); // Use data from front Fbo

                // Advance noise time by a visually pleasing step; wrap around if we run for waaaaay too long.
                double step = 0.005d;
                double seed = renderer.SimTime * step % 3e6d;
                m_noiseEffect.TimeStepUniform(new Vector2((float)seed, (float)step));
                m_noiseEffect.VarianceUniform(NoiseIntensityCoefficient);

                m_quad.Draw();
            }

            SwapBuffers();

            // more stuffs

            // The final scene should be left in the front buffer
        }

        #endregion
    }
}
