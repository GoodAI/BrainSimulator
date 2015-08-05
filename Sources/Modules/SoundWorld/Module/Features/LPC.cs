
using System;
namespace GoodAI.Modules.SoundProcessing.Features
{
    /// <summary>
    /// Exracts linear predictive coeficients. These are commonly used as feature vectors for speech recognition.
    /// </summary>
    public static class LPC
    {
        /// <summary>
        /// LPC Analysis using Durbin-Levinson's recursion algorithm
        /// </summary>
        public static float[] Compute(float[] x, int p) 
        {
            // Variable used in Durbin's algorithm
            float[] e = new float[p + 1];
            float[,] alpha = new float[p + 1, p + 1];
            float[] r = new float[p + 1];
            float[] k = new float[p + 1];
            float[] c = new float[p + 1];
            int N = x.Length;

            float[] lpc = new float[p + 1];

            // initialize
            for (int i = 0; i < p + 1; i++)
            {
                e[i] = k[i] = 0;
                for (int j = 0; j < p + 1; j++)
                    alpha[i, j] = 1;
            }

            // Autocorelation of input frame
            float max = 0;
            for (int i = 0; i < p + 1; i++)
            {
                r[i] = 0;
                for (int j = 0; j < N - i; j++)
                    r[i] += (x[j] * x[j + i]);

                if (Math.Abs(r[i]) > max)
                    max = Math.Abs(r[i]);
            }

            for (int i = 0; i < p + 1; i++)
            {
                r[i] /= max;
            }

            //float[] a = levinson(r);
            e[0] = r[0];

            // LPCAnalysis
            float sum;
            for (int i = 1; i <= p; i++)
            {
                sum = 0;
                for (int j = 1; j <= i - 1; j++)
                {
                    sum += (alpha[j,i - 1] * r[i - j]);
                }
                k[i] = (r[i] - sum) / e[i - 1];
                alpha[i,i] = k[i];
                for (int j = 1; j <= i - 1; j++)
                {
                    alpha[j,i] = alpha[j,i - 1] - k[i] * alpha[i - j,i - 1];
                }
                e[i] = (1 - k[i] * k[i]) * e[i - 1];
            }

            // extractSolution
            lpc[0] = 1;
            for (int i = 0; i < p; i++)
                lpc[i + 1] = -alpha[i + 1, p];

            //  1.0000   -1.1741    0.1443    0.1040    0.0754    0.0529    0.0370    0.0249    0.0144    0.0067    0.0351
            return lpc;
        }
    }
}
