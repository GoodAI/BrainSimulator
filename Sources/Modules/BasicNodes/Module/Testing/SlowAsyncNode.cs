using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.Testing
{
    /// <author>GoodAI</author>
    /// <meta>premek</meta>
    /// <status>Testing</status>
    /// <summary>Performs async CUDA computation.</summary>
    /// <description>
    /// Node intended for testing CUDA multi-streaming.
    /// </description>
    public sealed class SlowAsyncNode : MyWorkingNode
    {
        [MyInputBlock]
        public MyMemoryBlock<float> Input => GetInput(0);

        [MyNonpersistableOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value);}
        }

        public SlowTask Task { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = Input?.Count ?? 0;
        }
    }

    public class SlowTask : MyTask<SlowAsyncNode>
    {
        private MyCudaKernel m_polynomialKernel;

        public override void Init(int nGPU)
        {
            m_polynomialKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
        }

        public override void Execute()
        {
            m_polynomialKernel.Run(0f, 0f, 0f, 1f, Owner.Input, Owner.Output, Owner.Input.Count);
        }
    }
}
