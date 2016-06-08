using System;
using GoodAI.ToyWorld.Control;
using OpenTK.Graphics.OpenGL;
using Render.RenderObjects.Effects;
using RenderingBase.Renderer;
using VRageMath;
using World.ToyWorldCore;

namespace Render.RenderRequests
{
    internal class EffectRenderer
        : RRRendererBase<EffectSettings, RenderRequest>, IDisposable
    {
        #region Fields

        public const float AmbientTerm = 0.25f;

        protected SmokeEffect m_smokeEffect;
        protected PointLightEffect m_pointLightEffect;

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
            if (Settings.EnabledEffects.HasFlag(RenderRequestEffect.DayNight))
                return (1 - AmbientTerm) * world.Atlas.Day;

            return 1 - AmbientTerm;
        }


        #region Init

        public override void Init(RendererBase<ToyWorld> renderer, ToyWorld world, RenderRequest renderRequest, EffectSettings settings)
        {
            Settings = settings;

            if (Settings.EnabledEffects.HasFlag(RenderRequestEffect.Smoke))
            {
                if (m_smokeEffect == null)
                    m_smokeEffect = renderer.EffectManager.Get<SmokeEffect>();
                renderer.EffectManager.Use(m_smokeEffect); // Need to use the effect to set uniforms
                m_smokeEffect.SmokeColorUniform(new Vector4(Settings.SmokeColor.R, Settings.SmokeColor.G, Settings.SmokeColor.B, Settings.SmokeColor.A) / 255f);
            }

            if (Settings.EnabledEffects.HasFlag(RenderRequestEffect.Lights))
            {
                if (Settings.EnabledEffects.HasFlag(RenderRequestEffect.Lights) && m_pointLightEffect == null)
                    m_pointLightEffect = new PointLightEffect();
            }
        }

        #endregion

        #region Draw

        public override void Draw(RendererBase<ToyWorld> renderer, ToyWorld world)
        {
            // Set up transformation to world and screen space for noise effect
            Matrix mw = Matrix.Identity;
            // Model transform -- scale from (-1,1) to viewSize/2, center on origin
            mw *= Matrix.CreateScale(Owner.ViewV.Size / 2);
            // World transform -- move center to view center
            mw *= Matrix.CreateTranslation(new Vector3(Owner.ViewV.Center, 1f));
            // View and projection transforms
            Matrix mvp = mw * Owner.ViewProjectionMatrix;

            if (Settings.EnabledEffects.HasFlag(RenderRequestEffect.Lights))
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

                    Owner.Quad.Draw();
                }

                Owner.SetDefaultBlending();
            }

            if (Settings.EnabledEffects.HasFlag(RenderRequestEffect.Smoke))
            {
                renderer.EffectManager.Use(m_smokeEffect);
                m_smokeEffect.ModelWorldUniform(ref mw);
                m_smokeEffect.ModelViewProjectionUniform(ref mvp);

                m_smokeEffect.AmbientDiffuseTermsUniform(new Vector2(AmbientTerm, GetGlobalDiffuseComponent(world)));

                // Advance noise time by a visually pleasing step; wrap around if we run for waaaaay too long.
                double step = 0.005d * Settings.SmokeTransformationSpeedCoefficient;
                double seed = renderer.SimTime * step % 3e6d;
                m_smokeEffect.TimeStepUniform(new Vector2((float)seed, (float)step));
                m_smokeEffect.MeanScaleUniform(new Vector2(Settings.SmokeIntensityCoefficient, Settings.SmokeScaleCoefficient));

                Owner.Quad.Draw();
            }

            // more stufffs
        }

        #endregion
    }
}
