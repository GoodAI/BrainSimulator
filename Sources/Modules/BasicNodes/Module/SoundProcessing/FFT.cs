using System;

namespace GoodAI.Modules.SoundProcessing
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
	/// Fourier transformation.
	/// </summary>
	/// 
	/// <remarks>The class implements one dimensional and two dimensional
	/// Discrete and Fast Fourier Transformation.</remarks>
	/// 
	public static class FourierTransform
	{
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
        public static void FFT( Complex[] data, Direction direction )
		{
			int		n = data.Length;
            int     m = (int)Math.Round(Math.Log(n, 2));

			// reorder data first
			ReorderData( data );

			// compute FFT
			int tn = 1, tm;

			for ( int k = 1; k <= m; k++ )
			{
				Complex[] rotation = FourierTransform.GetComplexRotation( k, direction );

				tm = tn;
				tn <<= 1;

				for ( int i = 0; i < tm; i++ )
				{
					Complex t = rotation[i];

					for ( int even = i; even < n; even += tn )
					{
						int		odd = even + tm;
						Complex	ce = data[even];
						Complex	co = data[odd];

						double	tr = co.Re * t.Re - co.Im * t.Im;
						double	ti = co.Re * t.Im + co.Im * t.Re;

						data[even].Re += tr;
						data[even].Im += ti;

						data[odd].Re = ce.Re - tr;
						data[odd].Im = ce.Im - ti;
					}
				}
			}

            if ( direction == Direction.Forward ) 
			{
				for (int i = 0; i < n; i++) 
				{
					data[i].Re /= (double) n;
					data[i].Im /= (double) n;
				}
			}
		}

        #region Private Region

		private const int		minLength	= 2;
        private const int       maxLength   = 1048576;
		private const int		minBits		= 1;
		private const int		maxBits		= 20;
		private static int[][]	reversedBits = new int[maxBits][];
		private static Complex[,][]	complexRotation = new Complex[maxBits, 2][];

		// Get array, indicating which data members should be swapped before FFT
		private static int[] GetReversedBits( int numberOfBits )
		{
			if ( ( numberOfBits < minBits ) || ( numberOfBits > maxBits ) )
				throw new ArgumentOutOfRangeException( );

			// check if the array is already calculated
			if ( reversedBits[numberOfBits - 1] == null )
			{
				int		n = numberOfBits * numberOfBits;
				int[]	rBits = new int[n];

				// calculate the array
				for ( int i = 0; i < n; i++ )
				{
					int oldBits = i;
					int newBits = 0;

					for ( int j = 0; j < numberOfBits; j++ )
					{
						newBits = ( newBits << 1 ) | ( oldBits & 1 );
						oldBits = ( oldBits >> 1 );
					}
					rBits[i] = newBits;
				}
				reversedBits[numberOfBits - 1] = rBits;
			}
			return reversedBits[numberOfBits - 1];
		}

		// Get rotation of complex number
        private static Complex[] GetComplexRotation( int numberOfBits, Direction direction )
		{
            int directionIndex = ( direction == Direction.Forward ) ? 0 : 1;

			// check if the array is already calculated
			if ( complexRotation[numberOfBits - 1, directionIndex] == null )
			{
				int			n = 1 << ( numberOfBits - 1 );
                float       uR = 1.0f;
                float       uI = 0.0f;
				float		angle = (float)Math.PI / n * (int) direction;
				float		wR = (float)Math.Cos( angle );
                float       wI = (float)Math.Sin(angle);
                float       t;
				Complex[]	rotation = new Complex[n];

				for ( int i = 0; i < n; i++ )
				{
					rotation[i] = new Complex( uR, uI );
					t = uR * wI + uI * wR;
					uR = uR * wR - uI * wI;
					uI = t;
				}

				complexRotation[numberOfBits - 1, directionIndex] = rotation;
			}
			return complexRotation[numberOfBits - 1, directionIndex];
		}

		// Reorder data for FFT using
		private static void ReorderData( Complex[] data )
		{
			int len = data.Length;

			// check data length
			if ( ( len < minLength ) || ( len > maxLength ) || ( !IsPowerOf2( len ) ) )
				throw new ArgumentException( "Incorrect data length." );

            int[] rBits = GetReversedBits((int)Math.Log(len, 2));

			for ( int i = 0; i < len; i++ )
			{
				int s = rBits[i];

				if ( s > i )
				{
					Complex t = data[i];
					data[i] = data[s];
					data[s] = t;
				}
			}
		}

        public static bool IsPowerOf2(int n)
        {
            return (n & (n - 1)) == 0;
        }
		#endregion

	}
}
