using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    public class RawFloatBufferWriter : AbstractBufferReaderWriter, IRawBufferWriter
    {
        protected override void SetTypedBuffer(Array array)
        {
            buffer = (float[])array;
        }

        private float[] buffer;

        /// <summary>
        /// The user must set Buffer before use.
        /// </summary>
        public RawFloatBufferWriter()
        { }

        public RawFloatBufferWriter(float[] externalBuffer)
        {
            if (externalBuffer == null)
                throw new ArgumentNullException();

            Buffer = externalBuffer;
        }

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

    public class BufferWriter : IBufferWriter
    {
        protected IRawBufferWriter rawWriter;

        public BufferWriter(IRawBufferWriter rawWriter)
        {
            this.rawWriter = rawWriter;
        }
        
        #region IBuffer Writer implementations
        
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
}
