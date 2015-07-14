using System;
using System.Drawing;

namespace GoodAI.Modules.SoundProcessing.Features
{
    /// <summary>
    /// Exracts mel-frequency cepstral coeficients. These are commonly used as feature vectors for speech recognition.
    /// </summary>
    public static class MFCC
    {
        /// <summary>
        /// Compute mel-frequency cepstral coeficients for given frequency spectrum.
        /// </summary>
        /// <param name="fft">Frequency spectrum.</param>
        /// <param name="format">Format of input aoudio.</param>
        /// <param name="coefCount">Number of coeficients to compute.</param>
        /// <returns>Mel-frequency cepstral coeficients.</returns>
        public static float[] Compute(float[] fft, WaveFormat format, int coefCount)
        {
            if (format == null || fft == null || fft.Length == 0)
                return new float[0];

            // Compute mel scale filterbank
            float[] mel_scale = MelFilterBank(fft, coefCount, format.nSamplesPerSec);

            // Apply logarithm
            for (int i = 0; i < mel_scale.Length; i++)
                if (mel_scale[i] == 0)
                    continue;
                else
                    mel_scale[i] = (float)Math.Log10(mel_scale[i]);

            float[] mfcc = CosinusTransform(mel_scale);

            return mfcc;
        }

        private static float[] MelFilterBank(float[] fft, int filters_cnt, int sample_rate)
        {
            float[] mel_scale = new float[filters_cnt];

            // parametre pre (uzitocnu) polovicu frekvencneho spektra z FFT
            int fvz_pol = sample_rate / 2;                                                          // 8000
            int fft_pol = fft.Length / 2;                                                           // 256
            int m_step = (int)(MFCC.toMel((float)sample_rate) / ((float)(filters_cnt / 2) + 0.5f)); // 795
            int m_step_pol = m_step / 2;                                                            // 398

            // find peaks of triangular filters
            for (int n = 1; n < filters_cnt; n++)
            {
                // triangle filter in mel scale and convert to non-transformed scale
                int C = (int)MFCC.fromMel((n * m_step_pol) + m_step_pol) / fft_pol;
                int A = (int)MFCC.fromMel((n * m_step_pol) - m_step_pol) / fft_pol;
                int B = A + ((C - A) / 2);
                // compute weights of triangle filter
                float[] ab = Line(new Point(A, 0), new Point(B, 1));
                float[] bc = Line(new Point(B, 1), new Point(C, 0));
                // apply triangle fitler
                for (int i = A; i < B; i++)
                    mel_scale[n - 1] += fft[i] * ab[i - A];
                for (int i = B; i < C; i++)
                    mel_scale[n - 1] += fft[i] * bc[i - B];
            }

            return mel_scale;
        }

        private static float[] CosinusTransform(float[] input)
        {
            float sqrtOfLength = (float)Math.Sqrt(input.Length);

            int N = input.GetLength(0);

            float[] output = new float[N];

            for (int u = 0; u <= N - 1; u++)
            {
                float sum = 0.0f;
                for (int x = 0; x <= N - 1; x++)
                    sum += input[x] * (float)Math.Cos(((2.0 * x + 1.0) / (2.0 * N)) * u * Math.PI);
                output[u] = (float)Math.Round(sum);
            }
            return output;
        }

        /// <summary>
        /// Compute all samples of the line from point A to point B
        /// </summary>
        /// <param name="A">Left point</param>
        /// <param name="B">Right point</param>
        /// <returns>Array of line values</returns>
        private static float[] Line(Point A, Point B)
        {
            float[] res = new float[B.X - A.X];

            // y = a*x + b
            // a - gradient (slope)
            // b - Y intercept
            float a = (float)(B.Y - A.Y) / (float)(B.X - A.X);
            float b = A.Y - (a * A.X);

            for (int x = 0; x < res.Length; x++)
                res[x] = (a * (A.X + x) + b);

            return res;
        }

        private static float toMel(float f)
        {
            return 2595 * (float)Math.Log10(1 + f / 700);
        }

        private static float fromMel(float m)
        {
            return 700 * ((float)Math.Exp(m / 1127) - 1);
        }
    }
}
