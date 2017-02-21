using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Core.Memory
{
    /// <summary>
    /// This interface allows to mark node properties that have nested memory blocks (they are created automatically).
    /// Input or output memory blocks are not supported.
    /// </summary>
    public interface IMemBlockOwner
    {
        // You should have a method to do UpdateMemoryBlocks (with any params)
        // And probably some Execute method(s) called from tasks
    }

    /// <summary>
    /// More strict version of IMemBlockOwner that requires the UpdateMemoryBlocks method.
    /// </summary>
    public interface IMemBlockOwnerUpdatable : IMemBlockOwner
    {
        void UpdateMemoryBlocks();
    }

    /// <summary>
    /// Allows to specify custom name prefix for nested memory blocks.
    /// </summary>
    public interface IMemBlockNamePrefix : IMemBlockOwner
    {
        string MemBlockNamePrefix { get; }
    }
}
