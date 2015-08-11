using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    public abstract class AbstractBufferReader : AbstractBufferReaderWriter
    {
        protected static int RoundToNearestInt(double d)
        {
            return Convert.ToInt32(Math.Round(d, MidpointRounding.AwayFromZero));
        }

        /// <summary>
        /// Reads a double, does not check buffer boundaries.
        /// </summary>
        /// <returns>A double value of the next item</returns>
        protected abstract double ReadDoubleUnchecked();

        public double ReadDouble()
        {
            CheckBufferSpace(1);

            return ReadDoubleUnchecked();
        }

        public int ReadInt()
        {
            return RoundToNearestInt(ReadDouble());  // NOTE: converts float to double unnecessarily
        }

        public bool ReadBool()
        {
            return Convert.ToBoolean(ReadInt());
        }
    }

    public class FloatBufferReader : AbstractBufferReader
    {
        public float[] Buffer
        {
            get { return buffer; }
            set
            {
                buffer = value;
                index = 0;
                untypedBufferRef = buffer;
            }
        }

        private float[] buffer;

        /// <summary>
        /// The user must set Buffer before use.
        /// </summary>
        public FloatBufferReader()
        {
        }

        protected override double ReadDoubleUnchecked()
        {
            return (double)buffer[index++];
        }
    }
}
