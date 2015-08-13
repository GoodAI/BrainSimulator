using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    /// <summary>
    /// Raw buffer writer for float array. Handles the type conversion.
    /// 
    /// Intended to be used only as a low level layer for higher level buffer writers.
    /// </summary>
    public class RawFloatBufferWriter : AbstractRawFloatBufferReaderWriter, IRawBufferWriter
    {
        /// <summary>
        /// The user must set Buffer before use.
        /// </summary>
        public RawFloatBufferWriter(): base()
        {}

        public RawFloatBufferWriter(float[] externalBuffer): base(externalBuffer)
        {}

        public void PutDoubleUnchecked(double d)
        {
            buffer[index++] = (float)d;
        }

        public void PutFloatUnchecked(float f)
        {
            buffer[index++] = f;
        }

        public void PutIntUnchecked(int i)
        {
            buffer[index++] = (float)i;
        }
    }

    /// <summary>
    /// Buffer writer that can write basic value types to a typed buffer.
    /// The type of the target array is determinded by the IRawBufferWriter implementation used.
    /// </summary>
    public class BufferWriter : IBufferWriter
    {
        protected IRawBufferWriter rawWriter;

        public BufferWriter(IRawBufferWriter rawWriter)
        {
            this.rawWriter = rawWriter;
        }
        
        #region IBufferWriter implementation
        
        public Array Buffer
        {
            get { return rawWriter.Buffer; }
            set { rawWriter.Buffer = value; }
        }

        public void PutDouble(double d)
        {
            rawWriter.CheckBufferSpace(1);
            rawWriter.PutDoubleUnchecked(d);
        }

        public void PutFloat(float f)
        {
            rawWriter.CheckBufferSpace(1);
            rawWriter.PutFloatUnchecked(f);
        }

        public void PutInt(int i)
        {
            rawWriter.CheckBufferSpace(1);
            rawWriter.PutIntUnchecked(i);
        }

        public void PutBool(bool b)
        {
            PutInt(Convert.ToInt32(b));
        }

        #endregion
    }
    
    /// <summary>
    /// Provides easy construction of buffer writers for concrete types.
    /// </summary>
    public static class BufferWriterFactory
    {
        public static BufferWriter GetFloatWriter()
        {
            return new BufferWriter(new RawFloatBufferWriter());
        }

        public static BufferWriter GetFloatWriter(float[] buffer)
        {
            return new BufferWriter(new RawFloatBufferWriter(buffer));
        }
    }

    /// <summary>
    /// Use this a base class for a buffer writer when you want to use the generic BufferWriterFactory<>
    /// 
    /// The new() constraint of the factory requires a parameterless constructor.
    /// </summary>
    public class ConstructableBufferWriter : BufferWriter
    {
        /// <summary>
        /// Please do not use this parameterless consturctor, use the other one or a factory.
        /// </summary>
        public ConstructableBufferWriter() : base(null)
        {
            throw new InvalidOperationException("Please do not use this parameterless consturctor, use the other one or a factory.");
        }

        public ConstructableBufferWriter(IRawBufferWriter rawWriter) : base(rawWriter) {}
    }

    /// <summary>
    /// Generic factory for any buffer writer derived from ConstructableBufferWriter.
    /// </summary>
    /// <typeparam name="TBufferWriter">Type of the buffer writer that will be constructed.</typeparam>
    public static class BufferWriterFactory<TBufferWriter> where TBufferWriter : ConstructableBufferWriter, new()
    {
        public static TBufferWriter GetFloatWriter()
        {
            return (TBufferWriter)Activator.CreateInstance(typeof(TBufferWriter), new RawFloatBufferWriter());
        }

        public static TBufferWriter GetFloatWriter(float[] buffer)
        {
            return (TBufferWriter)Activator.CreateInstance(typeof(TBufferWriter), new RawFloatBufferWriter(buffer));
        }
    }
}
