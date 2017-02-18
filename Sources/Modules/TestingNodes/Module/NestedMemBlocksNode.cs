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
    /// <summary>Common interface for the following two testing clases representing "nested nodes".</summary>
    public interface ITestingNestedNode : IMemBlockOwnerUpdatable
    {
        int InputCount { get; set; }

        void Run(MyMemoryBlock<float> input);
    }

    /// <summary>
    /// Test class with one nested node, its Run method copies data from an external mem. block to its nested node.
    /// </summary>
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

    /// <summary>
    /// Testing class with two nested nodes implementing the same interface as the above class
    /// (for testing switching of implementations).
    /// </summary>
    public class Fooer : ITestingNestedNode
    {
        // ReSharper disable once UnusedAutoPropertyAccessor.Local -- the setters are necessary there!
        private MyMemoryBlock<float> FooOne { get; set; }
        // ReSharper disable once UnusedAutoPropertyAccessor.Local
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

    /// <summary>Very simple testing class with one valid and one invalid nested memory block.</summary>
    public class Barer : IMemBlockOwner
    {
        [MyInputBlock]  // Should be ignored.
        private MyMemoryBlock<float> Source { get; set; }

        private MyMemoryBlock<float> Bar { get; set; }
    }

    /// <summary>The simplest possible class with nested memory block. Not marked with any interface.</summary>
    public class Bazer  // Unmarked
    {
        internal MyMemoryBlock<float> Baz { get; set; }
    }

    /// <summary>A class with nested memory block to test memory persistence.</summary>
    public class Persistor
    {
        [MyPersistable]
        private MyMemoryBlock<float> Vinyl { get; set; }

        private int m_step;

        public void UpdateMemoryBlocks()
        {
            if (Vinyl == null)
            {
                MyLog.ERROR.WriteLine($"{nameof(Vinyl)} is null!");
                return;
            }

            Vinyl.Dims = new TensorDimensions(100, 100);
        }

        public void Execute()
        {
            if (Vinyl == null || Vinyl.Count == 0)
                return;

            m_step++;

            // Just do something with a clear temporal pattern.
            Vinyl.SafeCopyToHost();
            Vinyl.Host[(m_step / 100) % Vinyl.Count] = (float) Math.Sqrt(m_step);
            Vinyl.SafeCopyToDevice();
        }
    }

    /// <summary>
    /// Testing node that exercises different options of nested memory blocks.
    /// </summary>
    // ReSharper disable once ClassNeverInstantiated.Global
    public class NestedMemBlocksNode : MyWorkingNode
    {
        // Memory blocks inside this instance are automatically added to the MemoryManager, because:
        // (1) It derives from IMemoryBlockOwner.
        // (2) It is assigned using a property initializer, so it happens before MyNode's (ancestor of all nodes) constructor is called.
        internal MemBlockCowboy MemBlockCowboy { get; } = new MemBlockCowboy("Cowboy");

        // BEWARE: This does not work because MyNodeInfo is looking for memory blocks in the property type not instance type.
        // (The interface IMemBlockOwner does not have any memory blocks.)
        internal IMemBlockOwner Ignored { get; } = new Barer();

        // Barer's nested memory blocks are automatically initialized as well. (It fullfills the above mentioned conditions.)
        internal Barer Barer { get; } = new Barer();
        
        // This property is found by NodeInfo, but there's no instance to be searched for memory blocks.
        // It is initialized manually in the constructor.
        internal MemBlockCowboy MemBlockCowboyLateInit { get; }

        // Manually intialized nested node to test switching different sub-types in BrainSim's design time.
        internal ITestingNestedNode PolymorphNestedNode { get; private set; }

        // A manually initialized nested node does not need to be marked by any interface nor attribute.
        // And it does not have to be a property, it can be just a field.
        private readonly Bazer m_bazer;

        // For testing persistence (save/load memory block contents).
        internal Persistor Persistor { get; }

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


        public NestedBlockTask Task { get; private set; }

        public NestedMemBlocksNode()
        {
            MemBlockCowboyLateInit = new MemBlockCowboy("LateInit");
            CreateNestedMemoryBlocks(MemBlockCowboyLateInit);

            m_bazer = new Bazer();
            CreateNestedMemoryBlocks(m_bazer);

            CreateNestedMemoryBlocks(new object());  // Should emit a warning and otherwise be ignored.

            UpdateNestedNodeType(m_nestedNodeType);

            Persistor = new Persistor();
            CreateNestedMemoryBlocks(Persistor);
        }

        public override void UpdateMemoryBlocks()
        {
            var count = Input?.Count ?? 0;

            MemBlockCowboy.UpdateMemoryBlocks(count);

            MemBlockCowboyLateInit.UpdateMemoryBlocks(count);

            PolymorphNestedNode.InputCount = count;
            PolymorphNestedNode.UpdateMemoryBlocks();

            m_bazer.Baz.Count = 3 * count;

            Persistor.UpdateMemoryBlocks();
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
            Owner.Persistor.Execute();
        }
    }
}
