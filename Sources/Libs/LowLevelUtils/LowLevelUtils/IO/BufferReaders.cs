using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    /// <summary>
    /// Raw buffer reader for float array. Handles the type conversion.
    /// 
    /// Intended to be used only as a low level layer for higher level buffer readers.
    /// </summary>
    public class RawFloatBufferReader : AbstractRawFloatBufferReaderWriter, IRawBufferReader
    {
        /// <summary>
        /// The user must set Buffer before use.
        /// </summary>
        public RawFloatBufferReader(): base()
        {}

        public RawFloatBufferReader(float[] externalBuffer): base(externalBuffer)
        {}

        public double ReadDoubleUnchecked()
        {
            return (double)buffer[index++];
        }

        public float ReadFloatUnchecked()
        {
            return buffer[index++];
        }

        public int ReadIntUnchecked()
        {
            return RoundToNearestInt(ReadFloatUnchecked());
        }

        protected static int RoundToNearestInt(float f)
        {
            return Convert.ToInt32(Math.Round(f, MidpointRounding.AwayFromZero));
        }
    }

    /// <summary>
    /// Buffer reader that can read basic value types from a typed buffer.
    /// The type of the target array is determinded by the IRawBufferReader implementation used.
    /// </summary>
    public class BufferReader : IBufferReader
    {
        protected IRawBufferReader rawReader;

        public BufferReader(IRawBufferReader rawReader)
        {
            this.rawReader = rawReader;
        }

        #region IBufferReader implementation

        public Array Buffer
        {
            get { return rawReader.Buffer; }
            set { rawReader.Buffer = value; }
        }

        public double ReadDouble()
        {
            rawReader.CheckBufferSpace(1);
            return rawReader.ReadDoubleUnchecked();
        }

        public float ReadFloat()
        {
            rawReader.CheckBufferSpace(1);
            return rawReader.ReadFloatUnchecked();
        }

        public int ReadInt()
        {
            rawReader.CheckBufferSpace(1);
            return rawReader.ReadIntUnchecked();
        }

        public bool ReadBool()
        {
            return Convert.ToBoolean(ReadFloat());
        }

        #endregion
    }

    /// <summary>
    /// Provides easy construction of buffer readers for concrete types.
    /// </summary>
    public static class BufferReaderFactory
    {
        public static BufferReader GetFloatReader()
        {
            return new BufferReader(new RawFloatBufferReader());
        }

        public static BufferReader GetFloatReader(float[] buffer)
        {
            return new BufferReader(new RawFloatBufferReader(buffer));
        }
    }
}
