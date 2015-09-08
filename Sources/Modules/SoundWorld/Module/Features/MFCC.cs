using System;

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
        /// <param name="sample_rate">Sample rate of input audio.</param>
        /// <param name="coefCount">Number of coeficients to compute.</param>
        /// <returns>Mel-frequency cepstral coeficients.</returns>
        public static float[] Compute(float[] fft, int sample_rate, int coefCount)
        {
            if (fft == null || fft.Length == 0)
                return new float[0];

            // Compute mel scale filterbank
            float[] mel_scale = MelFilterBank(fft, (int)Math.Ceiling(coefCount * 1.5), sample_rate);

            // Apply logarithm
            for (int i = 0; i < mel_scale.Length; i++)
                if (mel_scale[i] == 0)
                    continue;
                else
                    mel_scale[i] = (float)Math.Log(mel_scale[i]);

            float[] dct = DCT(mel_scale);
            float[] mfcc = new float[coefCount];
            Array.Copy(dct, mfcc, coefCount);

            return mfcc;
        }

        internal class Point
        {
            public int X { get; set; }
            public int Y { get; set; }

            public Point(int x, int y)
            {
                this.X = x;
                this.Y = y;
            }
        }

        private static float[] MelFilterBank(float[] fft, int filters_cnt, int sample_rate)
        {
            // Prepare initial values
            float mel_min = toMel(0);
            float mel_max = toMel(sample_rate);
            float mel_step = (mel_max - mel_min) / (filters_cnt + 1);
            double[] f = new double[filters_cnt + 2];
            float[] mel = new float[filters_cnt + 2];
            float[] hertz = new float[filters_cnt + 2];
            float[] mel_scale = new float[filters_cnt];

            // Prepare count +2 mel bins
            mel[0] = mel_min;
            for (int n = 1; n < filters_cnt + 2; n++)
            {
                mel[n] = mel[n - 1] + mel_step;
            }

            // Convert to hertz bins and to FFT bins
            int nfft = fft.Length;
            for (int n = 0; n < filters_cnt + 2; n++)
            {
                hertz[n] = fromMel(mel[n]);
                f[n] = Math.Floor((nfft) * hertz[n] / sample_rate);
            }

            // find peaks of triangular filters
            for (int n = 1; n < filters_cnt; n++)
            {
                int A = (int)f[n - 1];
                int B = (int)f[n];
                int C = (int)f[n + 1];
                // triangle filter in mel scale and convert to non-transformed scale
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

        /// <summary>
        /// Forward Discrete Cosine Transform.
        /// </summary>
        /// 
        public static float[] DCT(float[] data)
        {
            int N = data.Length;
            float[] result = new float[N];
            for (int k = 0; k < N; k++)
            {
                double sum = 0;
                for (int n = 0; n < N; n++)
                    sum += data[n] * Math.Cos(Math.PI / (2.0 * N) * (2.0 * n + 1.0) * k);
                result[k] = (float)(Math.Sqrt(2.0 / N) * sum);
            }

            data[0] = result[0] / (float)Math.Sqrt(2);
            for (int i = 1; i < N; i++)
                data[i] = result[i];

            return data;
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
            return 1125 * (float)Math.Log(1 + (f / 700));
        }

        private static float fromMel(float m)
        {
            return 700 * ((float)Math.Exp(m / 1125) - 1);
        }
        
    }
}
