
using System;
namespace GoodAI.Modules.SoundProcessing.Features
{
    /// <summary>
    /// Exracts linear predictive coeficients. These are commonly used as feature vectors for speech recognition.
    /// </summary>
    public static class LPC
    {
        /// <summary>
        /// Algorithm for generation of autocorelation LPC coeficients invented by
        /// N. Levinsonom in 1947, modified by J. Durbinom in 1959.
        /// </summary>
        /// <param name="x">PCM data.</param>
        /// <param name="p">Number of LPC coeficients.</param>
        /// <returns>LPC coeficients.</returns>
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

            e[0] = r[0];

            // LPC analysis
            float sum;
            for (int i = 1; i <= p; i++)
            {
                sum = 0;
                for (int j = 1; j <= i - 1; j++)
                    sum += (alpha[j, i - 1] * r[i - j]);
                
                k[i] = (r[i] - sum) / e[i - 1];
                alpha[i, i] = k[i];
                for (int j = 1; j <= i - 1; j++)
                {
                    alpha[j, i] = alpha[j, i - 1] - k[i] * alpha[i - j, i - 1];
                }
                e[i] = (1 - k[i] * k[i]) * e[i - 1];
            }

            // Compute LPC
            for (int i = 0; i < p; i++)
                lpc[i + 1] = alpha[i + 1, p];

            sum = 0;
            for (int i = 1; i < c.Length; i++)
            {
                sum = 0;
                for (int j = 1; j <= i - 1; j++)
                    sum += ((j / (float)i) * c[j] * lpc[i - j - 1]);
                
                c[i] = lpc[i - 1] + sum;
            }

            return c;
        }

        /// <summary>
        /// Algorithm for generation of LPC coeficients error.
        /// </summary>
        /// <param name="data">PCM data.</param>
        /// <param name="lpc">LPC coeficients.</param>
        /// <returns>Error vector of LPC.</returns>
        public static float[] Error(float[] data, float[] lpc)
        {
            if (data == null | lpc == null)
                return null;

            int n = data.Length;
            int m = lpc.Length;
            float[] lpc1 = new float[m + 1];
            Array.Copy(lpc, 0, lpc1, 1, m);
            lpc1[0] = 1;
            float[] error = new float[n];

            for (int e = 0; e < n; e++)
                for (int i = 0; i < m; i++)
                    error[e] += lpc1[i] * ((e - i >= 0) ? data[e - i] : 0);
            
            return error;
        }

        /// <summary>
        /// Reverse synthesis of PCM signal using LPC coeficients and LPC error.
        /// </summary>
        /// <param name="error">LPC error.</param>
        /// <param name="lpc">LPC coeficients.</param>
        /// <returns></returns>
        public static float[] Reverse(float[] error, float[] lpc)
        {
            if (error == null | lpc == null)
                return null;

            int e = error.Length;
            int m = lpc.Length;
            float[] data = new float[e];
            for (int n = 0; n < e; n++)
            {
                for (int i = 0; i < m; i++)
                    data[n] += lpc[i] * ((n - i - 1 >= 0) ? data[n - i - 1] : 0);
                data[n] = error[n] - data[n];
            }
            return data;
        }
    }
}
