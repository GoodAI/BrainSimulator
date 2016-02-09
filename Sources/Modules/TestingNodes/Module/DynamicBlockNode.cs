using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.TestingNodes.DynamicBlocks
{
    /// <summary>
    /// A node that reallocates the output block in each step to a random size and fills it with random data.
    /// </summary>
    public class DynamicOutputNode : MyWorkingNode
    {
        public Random Random { get; private set; }

        public DynamicOutputNode()
        {
            Random = new Random();
        }

        [MyOutputBlock(0), DynamicBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public DynamicOutputTask Task { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = 50;
        }

        public override void ReallocateMemoryBlocks()
        {
            base.ReallocateMemoryBlocks();

            if (Output.Reallocate(GetNewBufferSize()))
            {
                MyLog.INFO.WriteLine("Reallocated the output block");
            }
            else
            {
                MyLog.WARNING.WriteLine("Failed to reallocate");
            }
        }

        public int GetNewBufferSize()
        {
            return Random.Next(99) + 1;
        }
    }

    /// <summary>
    /// See DynamicOutputNode.
    /// </summary>
    [Description("Dynamic block task")]
    public class DynamicOutputTask : MyTask<DynamicOutputNode>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            for (int i = 0; i < Owner.Output.Count; i++)
                Owner.Output.Host[i] = Owner.Random.Next(100);

            MyLog.INFO.WriteLine("Source output count: " + Owner.Output.Count);
            MyLog.INFO.WriteLine("Source output sum: " + Owner.Output.Host.Sum());

            Owner.Output.SafeCopyToDevice();
        }
    }

    /// <summary>
    /// A node that takes dynamic input, reallocates the output to match the input, copies the data
    /// using the GPU and prints out control sums.
    /// </summary>
    public class DynamicInputNode : MyWorkingNode
    {
        [MyInputBlock(0), DynamicBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0), DynamicBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public DynamicInputTask Task { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = Input != null ? Input.Count : 0;
        }

        public override void ReallocateMemoryBlocks()
        {
            base.ReallocateMemoryBlocks();

            Output.Reallocate(Input.Count);
        }
    }

    /// <summary>
    /// See DynamicInputNode.
    /// </summary>
    [Description("Dynamic block task")]
    public class DynamicInputTask : MyTask<DynamicInputNode>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.Output.GetDevice(Owner.GPU).CopyToDevice(Owner.Input.GetDevicePtr(Owner.GPU));

            Owner.Input.SafeCopyToHost();
            MyLog.INFO.WriteLine("Destination input count: " + Owner.Input.Count);
            MyLog.INFO.WriteLine("Destination input sum: " + Owner.Input.Host.Sum());

            Owner.Output.SafeCopyToHost();
            MyLog.INFO.WriteLine("Destination output count: " + Owner.Output.Count);
            MyLog.INFO.WriteLine("Destination output sum: " + Owner.Output.Host.Sum());
        }
    }
}
