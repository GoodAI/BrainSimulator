using System;
using System.Collections.Generic;
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
                    new NoEffectTex())
                .Case<NoEffectOffset>(() =>
                    new NoEffectOffset());
        }


        public T Get<T>()
            where T : EffectBase
        {
            return m_effects.Switch<T>();
        }

        public void Use(EffectBase effect)
        {
            if (m_currentEffect == effect)
                return;

            effect.Use();
            m_currentEffect = effect;
        }
    }
}
