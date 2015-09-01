using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;

namespace GoodAI.Core.Nodes
{
    /// <author>GoodAI</author>
    /// <status>Working</status>
    /// <summary>A node for gating signals from two input branches based on the value in the third branch</summary>
    /// <description>
    /// The node let you mix two inputs together proportionally to value in the Weight input.
    /// </description>
    public class MyDataGate : MyWorkingNode
    {
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input1
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Input2
        {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> Weight
        {
            get { return GetInput(2); }
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = GetInputSize(0);
            Output.ColumnHint = GetInput(0) != null ? GetInput(0).ColumnHint : 1;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(GetInputSize(0) == GetInputSize(1), this, "Input sizes differs!");
        }

        public override string Description
        {
            get
            {
                return "Gate Inputs";
            }
        }

        public MyGateTask GateInputs { get; private set; }

        /// <summary>
        /// Performs the gating.
        /// </summary>
        [Description("Gate Inputs")]
        public class MyGateTask : MyTask<MyDataGate>
        {
            private MyCudaKernel m_kernel;

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "InterpolateFromMemBlock");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.Output.Count);
                m_kernel.Run(Owner.Input1, Owner.Input2, Owner.Output, Owner.Weight, Owner.Output.Count);
            }
        }
    }
}
