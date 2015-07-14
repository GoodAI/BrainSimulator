
using System;
namespace GoodAI.Modules.SoundProcessing.Features
{
    /// <summary>
    /// Exracts linear predictive coeficients. These are commonly used as feature vectors for speech recognition.
    /// </summary>
    public static class LPC
    {
        /// <summary>
        /// Computes linear predictive coeficients features.
        /// </summary>
        /// <param name="frame">Input frame buffer.</param>
        /// <param name="coefCount">Number of ceoficients to be extracted.</param>
        /// <returns>Final LPC coeficients.</returns>
        public static float[] Compute(float[] frame, int coefCount)
        {
            float[] win = HammingWindow(frame);
            // Apply window funciton and compute autocorelation on current frame
            float[] ac = AutoCorrelate(win, coefCount);

            return LPCAnalysis(ac, coefCount);
        }

        /// <summary>
        /// Performs calculations of the Hamming function for a given frame
        /// </summary>
        /// <param name="frame">Frame to be windowed</param>
        /// <returns>Windowed frame</returns>
        private static float[] HammingWindow(float[] frame)
        {
            for (int n = 0; n < frame.Length; n++)
                frame[n] = 0.54f - 0.46f * (float)Math.Cos(2 * Math.PI * n / (frame.Length - 1));

            return frame;
        }

        /// <summary>
        /// Auto-correlation method
        /// </summary>
        /// <param name="frame"></param>
        /// <param name="order"></param>
        /// <returns></returns>
        private static float[] AutoCorrelate(float[] frame, int coefCount)
        {
            float[] ac = new float[coefCount + 1];
            for (int i = 0; i < coefCount + 1; i++)
            {
                ac[i] = 0;
                for (int j = 0; j < coefCount - i; j++)
                    ac[i] += (frame[j] * frame[j + i]);
            }

            return ac;
        }

        /// <summary>
        /// LPC Analysis using Durbin-Levinson's algorithm
        /// </summary>
        private static float[] LPCAnalysis(float[] frame, int coefCount)
        {
            // init
            float[] lpc = new float[coefCount + 1];
            float[] err = new float[coefCount + 1];
            float[] parcor = new float[coefCount + 1];
            float[,] alpha = new float[coefCount + 1, coefCount + 1];
            
            err[0] = frame[0];
            for (int i = 1; i <= coefCount; i++)
            {
                float sum = 0;
                for (int j = 1; j <= i - 1; j++)
                    sum += (alpha[j, i - 1] * frame[i - j]);

                parcor[i] = (frame[i] - sum) / err[i - 1];
                alpha[i, i] = parcor[i];

                for (int j = 1; j <= i - 1; j++)
                    alpha[j, i] = alpha[j, i - 1] - parcor[i] * alpha[i - j, i - 1];

                err[i] = (1 - parcor[i] * parcor[i]) * err[i - 1];
            }

            // extract solution
            for (int i = 0; i < coefCount; i++)
                lpc[i] = alpha[i + 1, coefCount];

            return lpc;
        }
    }
}
