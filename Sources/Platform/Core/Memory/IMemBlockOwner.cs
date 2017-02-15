using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Memory
{
    /// <summary>
    /// Node members derived from this interface can have inner memory blocks (they are created automatically).
    /// Input or output memory blocks are not supported.
    /// </summary>
    public interface IMemBlockOwner
    {
        // You should have a method to do UpdateMemoryBlocks (with any params)
        // And probably some Execute method(s) called from tasks
    }

    /// <summary>
    /// TODO(Premek)
    /// </summary>
    public interface IMemBlockOwnerUpdatable : IMemBlockOwner
    {
        void UpdateMemoryBlocks();
    }
}
