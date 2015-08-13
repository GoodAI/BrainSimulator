using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    public abstract class AbstractRawBufferReaderWriter
    {
        protected Array untypedBufferRef = null;

        protected int index = 0;

        public Array Buffer
        {
            get { return untypedBufferRef; }
            set
            {
                untypedBufferRef = value;
                SetTypedBuffer(value);
                index = 0;
            }
        }

        protected abstract void SetTypedBuffer(Array array);

        public void CheckBufferSpace(int size)
        {
            if (untypedBufferRef == null)
                throw new InvalidOperationException("Somebody didn't bother to set the buffer before use.");

            if (index + size > untypedBufferRef.Length)
                throw new IndexOutOfRangeException(
                    "There is no space for " + size + " additional items in the buffer.");
        }
    }

    public abstract class AbstractRawFloatBufferReaderWriter : AbstractRawBufferReaderWriter
    {
        protected override void SetTypedBuffer(Array array)
        {
            buffer = (float[])array;
        }

        protected float[] buffer;

        /// <summary>
        /// The user must set Buffer before use.
        /// </summary>
        public AbstractRawFloatBufferReaderWriter()
        { }

        public AbstractRawFloatBufferReaderWriter(float[] externalBuffer)
        {
            if (externalBuffer == null)
                throw new ArgumentNullException();

            Buffer = externalBuffer;
        }
    }
}
