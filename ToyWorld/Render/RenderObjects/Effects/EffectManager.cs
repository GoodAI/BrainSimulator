using System;
using System.Collections.Generic;
using Render.RenderObjects.Shaders;
using Render.Tests.Effects;
using Utils.VRageRIP.Lib.Collections;

namespace Render.RenderObjects.Effects
{
    internal class EffectManager
    {
        private readonly TypeSwitch<EffectBase> m_effects = new TypeSwitch<EffectBase>();

        private EffectBase m_currentEffect;


        public EffectManager()
        {
            m_effects
                .Case<NoEffect>(() =>
                    new NoEffect())
                .Case<NoEffectTex>(() =>
                    new NoEffectTex());
        }


        public void Use<T>()
            where T : EffectBase
        {
            var effect = m_effects.Switch<T>();

            if (m_currentEffect == effect)
                return;

            effect.Use();
            m_currentEffect = effect;
        }
    }
}
