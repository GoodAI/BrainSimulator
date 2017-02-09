using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.TestingNodes
{
    public class MemBlockOwner : IMemBlockOwner
    {
        public string Name { get; }

        public MemBlockOwner(string name)
        {
            Name = name;
        }

        private MyMemoryBlock<float> NestedAffairs { get; set; }
        
        public void UpdateMemoryBlocks(int count)
        {
            if (NestedAffairs == null)
            {
                MyLog.WARNING.WriteLine($"{nameof(NestedAffairs)}::{Name} is null!");
                return;
            }

            NestedAffairs.Count = count;

            MyLog.INFO.WriteLine($"{nameof(NestedAffairs)}::{Name} count set to {count}.");
        }

        public void Run(MyMemoryBlock<float> input)
        {
            if (NestedAffairs.Count < input.Count)
            {
                MyLog.ERROR.WriteLine($"{nameof(MemBlockOwner)}: Destination mem block too small: {NestedAffairs.Count}");
                return;
            }

            NestedAffairs.CopyFromMemoryBlock(input, 0, 0, input.Count);
        }
    }


    public class NestedMemBlocksNode : MyWorkingNode
    {
        internal MemBlockOwner MemBlockCowboy { get; } = new MemBlockOwner("Cowboy");
  
        internal MemBlockOwner MemBlockOwnerLateInit { get; }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input => GetInput(0);

        public NestedBlockTask Task { get; private set; }

        public NestedMemBlocksNode()
        {
            MemBlockOwnerLateInit = new MemBlockOwner("LateInit");

            CreateNestedMemoryBlocks(MemBlockOwnerLateInit);

            CreateNestedMemoryBlocks(new object());  // Should emit a warning and otherwise be ignored.
        }

        public override void UpdateMemoryBlocks()
        {
            MemBlockCowboy.UpdateMemoryBlocks(Input?.Count ?? 0);
            MemBlockOwnerLateInit.UpdateMemoryBlocks(Input?.Count ?? 0);
        }
    }

    [Description("Run sub-node"), MyTaskInfo]
    public class NestedBlockTask : MyTask<NestedMemBlocksNode>
    {
        public override void Init(int nGPU)
        {
            
        }

        public override void Execute()
        {
            Owner.MemBlockCowboy.Run(Owner.Input);
            Owner.MemBlockOwnerLateInit.Run(Owner.Input);
        }
    }
}
