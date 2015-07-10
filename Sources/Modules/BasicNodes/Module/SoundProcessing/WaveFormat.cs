using System.Runtime.InteropServices;

namespace AudioLib
{
    [StructLayout(LayoutKind.Sequential)]
    public class WaveFormat
    {
        public short wFormatTag;
        public short nChannels;
        public int nSamplesPerSec;
        public int nAvgBytesPerSec;
        public short nBlockAlign;
        public short wBitsPerSample;
        public short cbSize;

        /// <summary>
        /// Format of wave file.
        /// </summary>
        /// <param name="rate">Samples count per second.</param>
        /// <param name="bits">Bits count per sample.</param>
        /// <param name="channels">Number of channels.</param>
        public WaveFormat(int rate, int bits, int channels)
        {
            wFormatTag = 1;
            nChannels = (short)channels;
            nSamplesPerSec = rate;
            wBitsPerSample = (short)bits;
            cbSize = 0;

            nBlockAlign = (short)(channels * (bits / 8));
            nAvgBytesPerSec = nSamplesPerSec * nBlockAlign;
        }
    }//end class
}//end namespace
