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
    public interface ITestingNestedNode : IMemBlockOwnerUpdatable
    {
        int InputCount { get; set; }

        void Run(MyMemoryBlock<float> input);
    }

    public class MemBlockCowboy : ITestingNestedNode
    {
        private string Name { get; }

        public MemBlockCowboy(string name)
        {
            Name = name;
        }

        private MyMemoryBlock<float> NestedAffairs { get; set; }

        public int InputCount { get; set; }

        public void UpdateMemoryBlocks(int count)
        {
            if (NestedAffairs == null)
            {
                MyLog.WARNING.WriteLine($"{nameof(NestedAffairs)}::{Name} is null!");
                return;
            }

            NestedAffairs.Count = count;
            NestedAffairs.Name = $"{Name}:{nameof(NestedAffairs)}";

            MyLog.INFO.WriteLine($"{nameof(NestedAffairs)}::{Name} count set to {count}.");
        }

        public void UpdateMemoryBlocks()
        {
            UpdateMemoryBlocks(InputCount);
        }

        public void Run(MyMemoryBlock<float> input)
        {
            if (NestedAffairs.Count < input.Count)
            {
                MyLog.ERROR.WriteLine($"{nameof(MemBlockCowboy)}: Destination mem block too small: {NestedAffairs.Count}");
                return;
            }

            NestedAffairs.CopyFromMemoryBlock(input, 0, 0, input.Count);
        }
    }

    public class Fooer : ITestingNestedNode
    {
        private MyMemoryBlock<float> FooOne { get; set; }
        private MyMemoryBlock<float> FooTwo { get; set; }

        public void UpdateMemoryBlocks()
        {
            FooOne.Count = 10;
            FooTwo.Count = 20;
        }

        public void Run(MyMemoryBlock<float> input)
        {
            // No-op.
        }

        public int InputCount { get; set; }
    }

    public class Barer : IMemBlockOwner
    {
        [MyInputBlock]  // Should be ignored.
        private MyMemoryBlock<float> Source { get; set; }

        private MyMemoryBlock<float> Bar { get; set; }
    }

    public class Bazer  // Unmarked
    {
        public MyMemoryBlock<float> Baz { get; set; }
    }


    // ReSharper disable once ClassNeverInstantiated.Global
    public class NestedMemBlocksNode : MyWorkingNode
    {
        internal MemBlockCowboy MemBlockCowboy { get; } = new MemBlockCowboy("Cowboy");

        // BEWARE: This does not work because MyNodeInfo is looking for memory blocks in the property type not instance type.
        internal IMemBlockOwner Ignored { get; } = new Barer();

        internal Barer Barer { get; } = new Barer();

        internal MemBlockCowboy MemBlockCowboyLateInit { get; }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input => GetInput(0);

        public enum NestedNodeTypeEnum
        {
            Cowboy,
            Fooer
        }

        [MyBrowsable]
        public NestedNodeTypeEnum NestedNodeType
        {
            get{ return m_nestedNodeType; }
            set
            {
                if (value != m_nestedNodeType)
                    UpdateNestedNodeType(value);

                m_nestedNodeType = value;
            }
        }
        private NestedNodeTypeEnum m_nestedNodeType = NestedNodeTypeEnum.Cowboy;

        internal ITestingNestedNode PolymorphNestedNode { get; private set; }

        private Bazer Bazer { get; set; }

        public NestedBlockTask Task { get; private set; }

        public NestedMemBlocksNode()
        {
            MemBlockCowboyLateInit = new MemBlockCowboy("LateInit");
            CreateNestedMemoryBlocks(MemBlockCowboyLateInit);

            Bazer = new Bazer();
            CreateNestedMemoryBlocks(Bazer);
            Bazer.Baz.Count = 10;

            CreateNestedMemoryBlocks(new object());  // Should emit a warning and otherwise be ignored.

            UpdateNestedNodeType(m_nestedNodeType);
        }

        public override void UpdateMemoryBlocks()
        {
            var count = Input?.Count ?? 0;

            MemBlockCowboy.UpdateMemoryBlocks(count);

            MemBlockCowboyLateInit.UpdateMemoryBlocks(count);

            PolymorphNestedNode.InputCount = count;
            PolymorphNestedNode.UpdateMemoryBlocks();
        }

        private void UpdateNestedNodeType(NestedNodeTypeEnum nestedNodeType)
        {
            ITestingNestedNode newNestedNode;

            if (nestedNodeType == NestedNodeTypeEnum.Cowboy)
            {
                newNestedNode = new MemBlockCowboy("Poly");
            }
            else if (nestedNodeType == NestedNodeTypeEnum.Fooer)
            {
                newNestedNode = new Fooer();
            }
            else
            {
                throw new ArgumentOutOfRangeException(nameof(nestedNodeType), "Invalid nested node type.");
            }

            if (PolymorphNestedNode != null)
                DestroyNestedMemoryBlocks(PolymorphNestedNode);

            CreateNestedMemoryBlocks(newNestedNode);

            PolymorphNestedNode = newNestedNode;
        }
    }

    // ReSharper disable once ClassNeverInstantiated.Global
    [Description("Run sub-node"), MyTaskInfo]
    public class NestedBlockTask : MyTask<NestedMemBlocksNode>
    {
        public override void Init(int nGPU)
        {
            
        }

        public override void Execute()
        {
            Owner.MemBlockCowboy.Run(Owner.Input);
            Owner.MemBlockCowboyLateInit.Run(Owner.Input);

            Owner.PolymorphNestedNode.Run(Owner.Input);
        }
    }
}
