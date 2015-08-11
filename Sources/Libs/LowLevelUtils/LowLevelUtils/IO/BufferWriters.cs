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

    public class FloatBufferWriter : AbstractBufferWriter
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
}
