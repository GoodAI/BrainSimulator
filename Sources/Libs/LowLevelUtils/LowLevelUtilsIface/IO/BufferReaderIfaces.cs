using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.LowLevelUtils.IO
{
    /// <summary>
    /// Low level buffer reader that deals with conversion to from the type (presumably float or double)
    /// </summary>
    public interface IRawBufferReader
    {
        Array Buffer { get; set; }

        void CheckBufferSpace(int size);

        double ReadDoubleUnchecked();

        float ReadFloatUnchecked();

        int ReadIntUnchecked();
    }

    /// <summary>
    /// A buffer reader interface (should use IRawBufferReader to stay independent on the type of the array)
    /// </summary>
    public interface IBufferReader
    {
        Array Buffer { get; set; }

        double ReadDouble();

        float ReadFloat();

        int ReadInt();

        bool ReadBool();
    }
}
