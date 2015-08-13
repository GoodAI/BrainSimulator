using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    /// <summary>
    /// Low level buffer writer that deals with conversion to the destination type (presumably float or double)
    /// </summary>
    public interface IRawBufferWriter
    {
        Array Buffer { get; set; }

        void CheckBufferSpace(int size);

        void PutDoubleUnchecked(double d);

        void PutFloatUnchecked(float f);

        void PutIntUnchecked(int i);
    }
    
    /// <summary>
    /// A buffer writer interface (should use IRawBufferWriter to stay independent on the type of the array)
    /// </summary>
    public interface IBufferWriter
    {
        Array Buffer { get; set; }

        void PutDouble(double d);

        void PutFloat(float f);

        void PutInt(int i);

        void PutBool(bool i);
    }
}
