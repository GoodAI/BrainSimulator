using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public class EffectSettings
    {
        /// <summary>
        /// Specifies, if the day progression should dim the scene.
        /// </summary>
        public bool EnableDayAndNightCycle { get; set; }

        /// <summary>
        /// Specifies whether light sources should emanate light
        /// </summary>
        public bool DrawLights { get; set; }


        /// <summary>
        /// 
        /// </summary>
        public bool DrawSmoke { get; set; }

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


        public EffectSettings()
        { }
    }
}
