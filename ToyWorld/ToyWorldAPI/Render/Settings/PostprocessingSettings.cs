using System;
using System.Drawing;

namespace GoodAI.ToyWorld.Control
{
    [Flags]
    public enum RenderRequestPostprocessing
    {
        None,

        /// <summary>
        /// Adds random gaussian noise to every color component of the resulting scene.
        /// </summary>
        Noise,
    }

    /// <summary>
    /// 
    /// </summary>
    public class PostprocessingSettings
    {
        /// <summary>
        /// Specifies which postprocessing effects should be used.
        /// </summary>
        public RenderRequestPostprocessing EnabledPostprocessing { get; set; }


        #region Noise

        /// <summary>
        /// 
        /// </summary>
        public float NoiseIntensityCoefficient { get; set; }

        #endregion
    }
}
