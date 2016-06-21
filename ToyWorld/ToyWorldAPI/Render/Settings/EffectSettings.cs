using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    [Flags]
    public enum RenderRequestEffect
    {
        None,

        /// <summary>
        /// Specifies, if the day progression should dim the scene.
        /// </summary>
        DayNight = 1,
        /// <summary>
        /// Specifies whether light sources should emanate light
        /// </summary>
        Lights = 1 << 1,
        /// <summary>
        /// Draws cloud-like smoke above the world.
        /// </summary>
        Smoke = 1 << 2,
    }

    /// <summary>
    /// 
    /// </summary>
    public struct EffectSettings
    {
        /// <summary>
        /// Specifies which effects should be used.
        /// </summary>
        public RenderRequestEffect EnabledEffects { get; set; }


        #region Smoke settings

        /// <summary>
        /// 
        /// </summary>
        public Color SmokeColor { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public float SmokeTransformationSpeedCoefficient { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public float SmokeIntensityCoefficient { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public float SmokeScaleCoefficient { get; set; }

        #endregion


        public EffectSettings(RenderRequestEffect enabledEffects)
            : this()
        {
            EnabledEffects = enabledEffects;

            SmokeColor = Color.FromArgb(242, 242, 242, 242);
            SmokeTransformationSpeedCoefficient = 1f;
            SmokeIntensityCoefficient = 1f;
            SmokeScaleCoefficient = 1f;
        }
    }
}
