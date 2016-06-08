using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    /// <summary>
    /// 
    /// </summary>
    public class PostprocessingSettings
    {
        /// <summary>
        /// 
        /// </summary>
        public bool DrawNoise { get; set; }

        /// <summary>
        /// 
        /// </summary>
        public float NoiseIntensityCoefficient { get; set; }


        public PostprocessingSettings()
        { }
    }
}
