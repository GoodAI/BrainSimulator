using System;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Effects;
using RenderingBase.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    public class EffectRenderer
        : IDisposable
    {
        #region Fields

        public const float AmbientTerm = 0.25f;

        protected SmokeEffect m_smokeEffect;
        protected PointLightEffect m_pointLightEffect;


        private EffectSettings m_settings;

        #endregion

        #region Genesis

        public virtual void Dispose()
        {
            if (m_smokeEffect != null)
                m_smokeEffect.Dispose();
            if (m_pointLightEffect != null)
                m_pointLightEffect.Dispose();
        }

        #endregion


        public float GetGlobalDiffuseComponent(ToyWorld world)
        {
            if (m_settings.EnabledEffects.HasFlag(RenderRequestEffect.DayNight))
                return (1 - AmbientTerm) * world.Atlas.Day;

            return 1 - AmbientTerm;
        }


        #region Init

        public virtual void Init(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world, EffectSettings settings)
        {
            m_settings = settings;

            if (m_settings.EnabledEffects.HasFlag(RenderRequestEffect.Smoke))
            {
                if (m_smokeEffect == null)
                    m_smokeEffect = renderer.EffectManager.Get<SmokeEffect>();
                renderer.EffectManager.Use(m_smokeEffect); // Need to use the effect to set uniforms
                m_smokeEffect.SmokeColorUniform(new Vector4(m_settings.SmokeColor.R, m_settings.SmokeColor.G, m_settings.SmokeColor.B, m_settings.SmokeColor.A) / 255f);
            }

            if (m_settings.EnabledEffects.HasFlag(RenderRequestEffect.Lights))
            {
                if (m_settings.EnabledEffects.HasFlag(RenderRequestEffect.Lights) && m_pointLightEffect == null)
                    m_pointLightEffect = new PointLightEffect();
            }
        }

        #endregion

        #region Draw

        public virtual void Draw(RenderRequest renderRequest, RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // Set up transformation to world and screen space for noise effect
            Matrix mw = Matrix.Identity;
            // Model transform -- scale from (-1,1) to viewSize/2, center on origin
            mw *= Matrix.CreateScale(renderRequest.ViewV.Size / 2);
            // World transform -- move center to view center
            mw *= Matrix.CreateTranslation(new Vector3(renderRequest.ViewV.Center, 1f));
            // View and projection transforms
            Matrix mvp = mw * renderRequest.ViewProjectionMatrix;

            if (m_settings.EnabledEffects.HasFlag(RenderRequestEffect.Lights))
            {
                //GL.BlendFunc(BlendingFactorSrc.One, BlendingFactorDest.SrcAlpha); // Fades non-lit stuff to black
                GL.BlendFunc(BlendingFactorSrc.One, BlendingFactorDest.DstAlpha);

                // TODO: draw a smaller quad around the light source to minimize the number of framgent shader calls
                renderer.EffectManager.Use(m_pointLightEffect);
                m_pointLightEffect.ModelWorldUniform(ref mw);
                m_pointLightEffect.ModelViewProjectionUniform(ref mvp);

                foreach (var character in world.Atlas.Characters)
                {
                    m_pointLightEffect.ColorIntensityUniform(new Vector4(0.85f));
                    m_pointLightEffect.IntensityDecayUniform(new Vector2(1, character.ForwardSpeed));
                    m_pointLightEffect.LightPosUniform(new Vector3(character.Position));

                    renderRequest.Quad.Draw();
                }

                renderRequest.SetDefaultBlending();
            }

            if (m_settings.EnabledEffects.HasFlag(RenderRequestEffect.Smoke))
            {
                renderer.EffectManager.Use(m_smokeEffect);
                m_smokeEffect.ModelWorldUniform(ref mw);
                m_smokeEffect.ModelViewProjectionUniform(ref mvp);

                m_smokeEffect.AmbientDiffuseTermsUniform(new Vector2(AmbientTerm, GetGlobalDiffuseComponent(world)));

                // Advance noise time by a visually pleasing step; wrap around if we run for waaaaay too long.
                double step = 0.005d * m_settings.SmokeTransformationSpeedCoefficient;
                double seed = renderer.SimTime * step % 3e6d;
                m_smokeEffect.TimeStepUniform(new Vector2((float)seed, (float)step));
                m_smokeEffect.MeanScaleUniform(new Vector2(m_settings.SmokeIntensityCoefficient, m_settings.SmokeScaleCoefficient));

                renderRequest.Quad.Draw();
            }

            // more stufffs
        }

        #endregion
    }
}
