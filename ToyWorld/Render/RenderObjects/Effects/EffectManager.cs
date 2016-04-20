using System;
using System.Collections.Generic;
using Utils.VRageRIP.Lib.Collections;

namespace Render.RenderObjects.Effects
{
    internal class EffectManager
    {
        private readonly TypeSwitch<EffectBase> m_effects = new TypeSwitch<EffectBase>();

        private EffectBase m_currentEffect;


        public EffectManager()
        {
            CaseInternal<NoEffect>();
            CaseInternal<NoEffectTex>();
            CaseInternal<NoEffectOffset>();
        }

        private void CaseInternal<T>()
            where T : EffectBase, new()
        {
            m_effects.Case<T>(() => new T());
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
