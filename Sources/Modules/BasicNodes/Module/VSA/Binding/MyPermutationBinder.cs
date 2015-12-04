using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace GoodAI.Modules.VSA
{
    public class MyPermutationBinder : MySymbolBinderBase
    {
        private MyCudaKernel m_binaryPermKernel;

        CudaStream m_stream;


        public MyPermutationBinder(MyWorkingNode owner, int inputSize, MyMemoryBlock<float> tempBlock)
            : base(owner, inputSize, tempBlock)
        {
            m_stream = new CudaStream();

            m_binaryPermKernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
            m_binaryPermKernel.SetupExecution(inputSize);
        }

        public override void Bind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output)
        {
            Bind(firstInput, otherInputs.ToArray(), output, (int)MyJoin.MyJoinOperation.Permutation);
        }

        public override void Unbind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output)
        {
            Bind(firstInput, otherInputs.ToArray(), output, (int)MyJoin.MyJoinOperation.Inv_Permutation);
        }

        void Bind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output, int method)
        {
            if (otherInputs == null)
                throw new ArgumentNullException("otherInputs");


            var second = otherInputs.FirstOrDefault();

            if (second == null)
                throw new ArgumentException("Nothing to bind with...");


            m_binaryPermKernel.RunAsync(m_stream, firstInput, second, output, method, m_inputSize);

            foreach (var input in otherInputs.Skip(1)) // Exclude the second input
                m_binaryPermKernel.RunAsync(m_stream, input, output, output, method, m_inputSize);
        }
    }
}
