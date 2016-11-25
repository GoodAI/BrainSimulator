using Utils.VRageRIP.Lib.Collections;

namespace RenderingBase.RenderObjects.Effects
{
    public class EffectManager
    {
        private readonly TypeSwitch<EffectBase> m_effects = new TypeSwitch<EffectBase>();

        private EffectBase m_currentEffect;


        public void Case<T>()
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
