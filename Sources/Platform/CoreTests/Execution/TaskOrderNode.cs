using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;

namespace CoreTests.Execution
{
    /// <summary>
    /// A testing node that uses task ordering by attributes.
    /// </summary>
    internal sealed class TaskOrderNode : MyWorkingNode
    {
        public FooTask FooTaskProp { get; private set; }
        public BarTask BarTaskProp { get; private set; }
        public ZedTask ZedTaskProp { get; private set; }

        [MyTaskInfo(Order = 0)]
        public class ZedTask : MyTask<TaskOrderNode>
        {
            public override void Init(int nGPU) { }
            public override void Execute() { }
        }

        [MyTaskInfo(Order = 1)]
        public class FooTask : MyTask<TaskOrderNode>
        {
            public override void Init(int nGPU) { }
            public override void Execute() { }
        }

        [MyTaskInfo(Order = 2)]
        public class BarTask : MyTask<TaskOrderNode>
        {
            public override void Init(int nGPU) { }
            public override void Execute() { }
        }

        public override void UpdateMemoryBlocks()
        {
        }
    }
    
    /// <summary>
    /// A testing node that does NOT use task ordering.
    /// </summary>
    internal sealed class UnorderedTasksNode : MyWorkingNode
    {
        public CherryTask CherryTaskProp { get; private set; }
        public BananaTask BananaTaskProp { get; private set; }
        public AppleTask AppleTaskProp { get; private set; }

        public class AppleTask : MyTask<UnorderedTasksNode>
        {
            public override void Init(int nGPU) { }
            public override void Execute() { }
        }

        public class BananaTask : MyTask<UnorderedTasksNode>
        {
            public override void Init(int nGPU) { }
            public override void Execute() { }
        }

        public class CherryTask : MyTask<UnorderedTasksNode>
        {
            public override void Init(int nGPU) { }
            public override void Execute() { }
        }

        public override void UpdateMemoryBlocks()
        {
        }
    }
}
