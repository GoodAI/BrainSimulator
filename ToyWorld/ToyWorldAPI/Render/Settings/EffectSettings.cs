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
        DayNight,
        /// <summary>
        /// Specifies whether light sources should emanate light
        /// </summary>
        Lights,
        /// <summary>
        /// Draws cloud-like smoke above the world.
        /// </summary>
        Smoke,
    }

    /// <summary>
    /// 
    /// </summary>
    public class EffectSettings
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
    }
}
