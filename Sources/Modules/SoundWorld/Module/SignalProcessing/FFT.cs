using System;

namespace GoodAI.Modules.SoundProcessing
{
	/// <summary>
	/// Fourier transformation.
	/// </summary>
	/// 
	/// <remarks>The class implements one dimensional and two dimensional
	/// Discrete and Fast Fourier Transformation.</remarks>
	/// 
	public static class FourierTransform
	{
        /// <summary>
        /// Fourier transformation direction.
        /// </summary>
        public enum Direction
        {
            /// <summary>
            /// Forward direction of Fourier transformation.
            /// </summary>
            Forward = 1,

            /// <summary>
            /// Backward direction of Fourier transformation.
            /// </summary>
            Backward = -1
        };

        /// <summary>
        /// One dimensional Fast Fourier Transform.
        /// </summary>
        /// 
        /// <param name="data">Data to transform.</param>
        /// <param name="direction">Transformation direction.</param>
        /// 
        /// <remarks><para><note>The method accepts <paramref name="data"/> array of 2<sup>n</sup> size
        /// only, where <b>n</b> may vary in the [1, 14] range.</note></para></remarks>
        /// 
        /// <exception cref="ArgumentException">Incorrect data length.</exception>
        /// 
        public static void FFT(Complex[] data, Direction direction)
        {
            int n = data.Length;
            int m = Log2(n);

            // reorder data first
            ReorderData(data);

            // compute FFT
            int tn = 1, tm;

            for (int k = 1; k <= m; k++)
            {
                Complex[] rotation = FourierTransform.GetComplexRotation(k, direction);

                tm = tn;
                tn <<= 1;

                for (int i = 0; i < tm; i++)
                {
                    Complex t = rotation[i];

                    for (int even = i; even < n; even += tn)
                    {
                        int odd = even + tm;
                        Complex ce = data[even];
                        Complex co = data[odd];

                        double tr = co.Re * t.Re - co.Im * t.Im;
                        double ti = co.Re * t.Im + co.Im * t.Re;

                        data[even].Re += tr;
                        data[even].Im += ti;

                        data[odd].Re = ce.Re - tr;
                        data[odd].Im = ce.Im - ti;
                    }
                }
            }

            if (direction == Direction.Forward)
            {
                for (int i = 0; i < n; i++)
                {
                    data[i].Re /= (double)n;
                    data[i].Im /= (double)n;
                }
            }
        }

        #region Private Region

        private const int minLength = 2;
        private const int maxLength = 16384;
        private const int minBits = 1;
        private const int maxBits = 14;
        private static int[][] reversedBits = new int[maxBits][];
        private static Complex[,][] complexRotation = new Complex[maxBits, 2][];

        /// <summary>
        /// Calculates power of 2.
        /// </summary>
        /// 
        /// <param name="power">Power to raise in.</param>
        /// 
        /// <returns>Returns specified power of 2 in the case if power is in the range of
        /// [0, 30]. Otherwise returns 0.</returns>
        /// 
        public static int Pow2(int power)
        {
            return ((power >= 0) && (power <= 30)) ? (1 << power) : 0;
        }

        /// <summary>
        /// Checks if number is power of 2.
        /// </summary>
        /// <param name="n">Input number.</param>
        /// <returns>True if input is power of 2.</returns>
        public static bool IsPowerOf2(int n)
        {
            return (n & (n - 1)) == 0;
        }

        /// <summary>
        /// Get base of binary logarithm.
        /// </summary>
        /// 
        /// <param name="x">Source integer number.</param>
        /// 
        /// <returns>Power of the number (base of binary logarithm).</returns>
        /// 
        public static int Log2( int x )
        {
            if ( x <= 65536 )
            {
                if ( x <= 256 )
                {
                    if ( x <= 16 )
                    {
                        if ( x <= 4 )
                        {
                            if ( x <= 2 )
                            {
                                if ( x <= 1 )
                                    return 0;
                                return 1;
                            }
                            return 2;
                        }
                        if ( x <= 8 )
                            return 3;
                        return 4;
                    }
                    if ( x <= 64 )
                    {
                        if ( x <= 32 )
                            return 5;
                        return 6;
                    }
                    if ( x <= 128 )
                        return 7;
                    return 8;
                }
                if ( x <= 4096 )
                {
                    if ( x <= 1024 )
                    {
                        if ( x <= 512 )
                            return 9;
                        return 10;
                    }
                    if ( x <= 2048 )
                        return 11;
                    return 12;
                }
                if ( x <= 16384 )
                {
                    if ( x <= 8192 )
                        return 13;
                    return 14;
                }
                if ( x <= 32768 )
                    return 15;
                return 16;
            }

            if ( x <= 16777216 )
            {
                if ( x <= 1048576 )
                {
                    if ( x <= 262144 )
                    {
                        if ( x <= 131072 )
                            return 17;
                        return 18;
                    }
                    if ( x <= 524288 )
                        return 19;
                    return 20;
                }
                if ( x <= 4194304 )
                {
                    if ( x <= 2097152 )
                        return 21;
                    return 22;
                }
                if ( x <= 8388608 )
                    return 23;
                return 24;
            }
            if ( x <= 268435456 )
            {
                if ( x <= 67108864 )
                {
                    if ( x <= 33554432 )
                        return 25;
                    return 26;
                }
                if ( x <= 134217728 )
                    return 27;
                return 28;
            }
            if ( x <= 1073741824 )
            {
                if ( x <= 536870912 )
                    return 29;
                return 30;
            }
            return 31;
        }

        // Get array, indicating which data members should be swapped before FFT
        private static int[] GetReversedBits(int numberOfBits)
        {
            if ((numberOfBits < minBits) || (numberOfBits > maxBits))
                throw new ArgumentOutOfRangeException();

            // check if the array is already calculated
            if (reversedBits[numberOfBits - 1] == null)
            {
                int n = Pow2(numberOfBits);
                int[] rBits = new int[n];

                // calculate the array
                for (int i = 0; i < n; i++)
                {
                    int oldBits = i;
                    int newBits = 0;

                    for (int j = 0; j < numberOfBits; j++)
                    {
                        newBits = (newBits << 1) | (oldBits & 1);
                        oldBits = (oldBits >> 1);
                    }
                    rBits[i] = newBits;
                }
                reversedBits[numberOfBits - 1] = rBits;
            }
            return reversedBits[numberOfBits - 1];
        }

        // Get rotation of complex number
        private static Complex[] GetComplexRotation(int numberOfBits, Direction direction)
        {
            int directionIndex = (direction == Direction.Forward) ? 0 : 1;

            // check if the array is already calculated
            if (complexRotation[numberOfBits - 1, directionIndex] == null)
            {
                int n = 1 << (numberOfBits - 1);
                double uR = 1.0;
                double uI = 0.0;
                double angle = System.Math.PI / n * (int)direction;
                double wR = System.Math.Cos(angle);
                double wI = System.Math.Sin(angle);
                double t;
                Complex[] rotation = new Complex[n];

                for (int i = 0; i < n; i++)
                {
                    rotation[i] = new Complex(uR, uI);
                    t = uR * wI + uI * wR;
                    uR = uR * wR - uI * wI;
                    uI = t;
                }

                complexRotation[numberOfBits - 1, directionIndex] = rotation;
            }
            return complexRotation[numberOfBits - 1, directionIndex];
        }

        // Reorder data for FFT using
        private static void ReorderData(Complex[] data)
        {
            int len = data.Length;

            // check data length
            if ((len < minLength) || (len > maxLength) || (!IsPowerOf2(len)))
                throw new ArgumentException("Incorrect data length.");

            int[] rBits = GetReversedBits(Log2(len));

            for (int i = 0; i < len; i++)
            {
                int s = rBits[i];

                if (s > i)
                {
                    Complex t = data[i];
                    data[i] = data[s];
                    data[s] = t;
                }
            }
        }

        #endregion

	}
}
