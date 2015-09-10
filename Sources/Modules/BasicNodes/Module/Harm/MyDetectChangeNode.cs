using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Harm
{
    /// <author>GoodAI</author>
    /// <meta>jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Computes direction of change of the input variables (independently). 
    /// </summary>
    /// <description>
    /// The value on the output is given by parameter OutputScale and is zero/positive/negative. Used for producing rewards to the DiscreteQLearningNode.
    /// </description>
    public class MyDetectChangesNode : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> DataInput
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> DataOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public MyMemoryBlock<float> PrevInputs { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            if (DataInput != null)
            {
                DataOutput.Count = DataInput.Count;
                PrevInputs.Count = DataInput.Count;
            }
            else
            {
                DataOutput.Count = 1;
                PrevInputs.Count = 1;
            }
        }

        public MyDifferenceDetector DetectDifferences { get; private set; }

        /// <summary>
        /// Detect difference compared to previous input and publish it.
        /// </summary>
        [MyTaskInfo(OneShot = false)]
        public class MyDifferenceDetector : MyTask<MyDetectChangesNode>
        {

            [MyBrowsable, Category("IO"), DisplayName("Output Scale"),
            Description("Node computes one of these values for each element: {0, +OutputScale, -OutputScale}")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float RewardSize { get; set; }



            private MyCudaKernel m_kernel;
            private MyCudaKernel m_copyKernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Harm\MatrixQLearningKernel", "detectChanges");
                m_copyKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Harm\MatrixQLearningKernel", "copyKernel");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.DataInput.Count);
                m_copyKernel.SetupExecution(Owner.DataInput.Count);

                if (base.SimulationStep > 0)
                {
                    m_kernel.Run(Owner.DataInput, Owner.PrevInputs, Owner.DataOutput, Owner.DataInput.Count, RewardSize);
                }
                else
                {
                    Owner.DataOutput.Fill(0);
                }
                m_copyKernel.Run(Owner.DataInput, Owner.PrevInputs, Owner.DataInput.Count);
            }
        }
    }
}
