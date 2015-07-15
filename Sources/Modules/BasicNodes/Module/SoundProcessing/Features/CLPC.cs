
namespace GoodAI.Modules.SoundProcessing.Features
{
    /// <summary>
    /// Exracts cepstral linear predictive coeficients. These are commonly used as feature vectors for speech recognition.
    /// </summary>
    public static class CLPC
    {
        public static float[] Compute(float[] inputs, int coefCount)
        {
            // compute LPC coefs first
            float[] lpc = LPC.Compute(inputs, coefCount);

            // then calculate cepstral LPCs from LPC
            int i, j;
            float sum;
            float[] coefs = new float[coefCount];
            
            for (i = 1; i < coefCount; i++)
            {
                sum = 0;
                for (j = 1; j <= i - 1; j++)
                    sum += ((j / (float)i) * coefs[j] * lpc[i - j - 1]);
                
                coefs[i] = lpc[i - 1] + sum;
            }

            return coefs;
        }
    }
}
