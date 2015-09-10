using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using ManagedCuda.BasicTypes;

namespace GoodAI.Modules.VSA
{
    public class MyPermutationBinder : MySymbolBinderBase
    {
        private MyCudaKernel m_PermKernel, m_binaryPermKernel;


        public MyPermutationBinder(MyWorkingNode owner, int inputSize, MyMemoryBlock<float> tempBlock)
            : base(owner, inputSize, tempBlock)
        {
            m_PermKernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Common\CombineVectorsKernel", "CombineVectorsKernel");
            m_PermKernel.SetupExecution(inputSize);
            m_binaryPermKernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
            m_binaryPermKernel.SetupExecution(inputSize);
        }


        public override void Bind(CUdeviceptr firstInput, params CUdeviceptr[] otherInputs)
        {
            Bind(firstInput, otherInputs, (int)MyJoin.MyJoinOperation.Permutation);
        }

        public override void Unbind(CUdeviceptr firstInput, params CUdeviceptr[] otherInputs)
        {
            Bind(firstInput, otherInputs, (int)MyJoin.MyJoinOperation.Inv_Permutation);
        }

        void Bind(CUdeviceptr firstInput, CUdeviceptr[] otherInputs, int method)
        {
            if (otherInputs == null)
                otherInputs = new[] { firstInput };

            var output = otherInputs[otherInputs.Length - 1];


            if (otherInputs.Length <= 2)
            {
                m_binaryPermKernel.Run(firstInput, otherInputs[0], output, method, m_inputSize);
                return;
            }


            m_tempBlock.Host[0] = firstInput;

            for (int i = 1; i < otherInputs.Length; i++)
                m_tempBlock.Host[i] = otherInputs[i - 1];

            m_tempBlock.SafeCopyToDevice(0, otherInputs.Length);
            m_PermKernel.Run(m_tempBlock, m_tempBlock.Count, output, method, m_inputSize);
        }
    }
}
