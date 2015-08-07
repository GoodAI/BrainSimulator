
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
        /// <param name="data">PCM data.</param>
        /// <param name="m">Number of LPC coeficients.</param>
        /// <returns>LPC coeficients.</returns>
        public static float[] Compute(float[] data, int m)
        {
            float[] aut = new float[m + 1];
            float[] lpc = new float[m];
            float error;
            int n = data.Length;
            int i, j;

            // autocorrelation, p+1 lag coefficients
            j = m + 1;
            while (j-- != 0)
            {
                float d = 0.0F;
                for (i = j; i < n; i++) d += data[i] * data[i - j];
                aut[j] = d;
            }

            // Generate lpc coefficients from autocorr values
            error = aut[0];

            for (i = 0; i < m; i++)
            {
                float r = -aut[i + 1];
                if (error == 0)
                {
                    for (int k = 0; k < m; k++) lpc[k] = 0.0f;
                    return null;
                }

                // Sum up this iteration's reflection coefficient; note that in
                // Vorbis we don't save it.  If anyone wants to recycle this code
                // and needs reflection coefficients, save the results of 'r' from
                // each iteration.
                for (j = 0; j < i; j++) r -= lpc[j] * aut[i - j];
                r /= error;

                // Update LPC coefficients and total error
                lpc[i] = r;
                for (j = 0; j < i / 2; j++)
                {
                    float tmp = lpc[j];
                    lpc[j] += r * lpc[i - 1 - j];
                    lpc[i - 1 - j] += r * tmp;
                }
                if (i % 2 != 0) lpc[j] += lpc[j] * r;

                error *= (float)(1.0 - r * r);
            }

            // we need the error value to know how big an impulse to hit the
            // filter with later
            return lpc;
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
                {
                    error[e] += lpc1[i] * ((e - i >= 0) ? data[e - i] : 0);
                }
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
