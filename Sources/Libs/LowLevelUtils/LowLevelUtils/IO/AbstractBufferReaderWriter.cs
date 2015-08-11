using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    public abstract class AbstractBufferReaderWriter
    {
        /// <summary>
        /// The derived class must set this to reference the typed buffer (presumably float[] or double[])
        /// </summary>
        protected Array untypedBufferRef = null;

        protected int index = 0;

        protected void CheckBufferSpace(int size)
        {
            if (untypedBufferRef == null)
                throw new InvalidOperationException("Somebody didn't bother to set the buffer before use.");

            if (index + size > untypedBufferRef.Length)
                throw new IndexOutOfRangeException(
                    "There is no space for " + size + " additional items in the buffer.");
        }
    }
}
