using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.BrainSimulator.Utils
{
    public static class Profiling
    {
        public static readonly Color ColorAboveThreshold = Color.FromArgb(255, 255, 0, 0);
        public static readonly Color ColorBelowThreshold = Color.FromArgb(255, 0, 255, 0);

        public static readonly double ProfilingThreshold = 0.05;
        public static readonly double MinimalSaturation = 0.15;

        public static double ScaleSaturation(double factor, double minimalSaturation)
        {
            return minimalSaturation + factor - (factor*minimalSaturation);
        }

        public static Color ItemColor(double factor)
        {
            Color baseColor;
            if (factor > ProfilingThreshold)
            {
                baseColor = ColorAboveThreshold;
            }
            else
            {
                baseColor = ColorBelowThreshold;
                factor = ProfilingThreshold;
            }

            double scaledFactor = ScaleSaturation(factor, MinimalSaturation);

            return Color.FromArgb(255,
                Saturate(baseColor.R, scaledFactor),
                Saturate(baseColor.G, scaledFactor),
                Saturate(baseColor.B, scaledFactor));
        }

        private static int Saturate(int channel, double factor)
        {
            return (int) (channel + (255 - channel)*(1 - factor));
        }
    }
}
