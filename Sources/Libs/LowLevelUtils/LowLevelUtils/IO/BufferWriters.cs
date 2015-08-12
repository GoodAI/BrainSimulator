using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    public abstract class AbstractBufferWriter : AbstractBufferReaderWriter
    {
        protected abstract void PutDoubleUnchecked(double d);

        public void PutDouble(double d)
        {
            CheckBufferSpace(1);
            PutDoubleUnchecked(d);
        }

        protected abstract void PutFloatUnchecked(float f);

        public void PutFloat(float f)
        {
            CheckBufferSpace(1);
            PutFloatUnchecked(f);
        }

        public abstract void PutInt(int i);

        public void PutBool(bool b)
        {
            PutInt(Convert.ToInt32(b));
        }
    }

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

    public class FloatBufferWriter : AbstractBufferWriter
    {
        private float[] buffer;

        protected override void SetTypedBuffer(Array array)
        {
            buffer = (float[])array;
        }

        /// <summary>
        /// The user must set Buffer before use.
        /// </summary>
        public FloatBufferWriter()
        { }

        public FloatBufferWriter(float[] externalBuffer)
        {
            if (externalBuffer == null)
                throw new ArgumentNullException();

            Buffer = externalBuffer;
        }

        protected override void PutDoubleUnchecked(double d)
        {
            buffer[index++] = (float)d;
        }

        protected override void PutFloatUnchecked(float f)
        {
            buffer[index++] = f;
        }

        public override void PutInt(int i)
        {
            CheckBufferSpace(1);

            buffer[index++] = (float)i;
        }
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
