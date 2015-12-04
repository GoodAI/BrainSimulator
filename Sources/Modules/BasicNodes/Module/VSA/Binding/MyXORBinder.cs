using System;
using System.Collections.Generic;
using System.Linq;
using GoodAI.Core;
using GoodAI.Core.Nodes;
using ManagedCuda.BasicTypes;

namespace GoodAI.Modules.VSA
{
    public class MyXORBinder : MySymbolBinderBase
    {
        private MyCudaKernel m_XORKernel;


        public MyXORBinder(MyWorkingNode owner, int inputSize)
            : base(owner, inputSize, null)
        {
            m_XORKernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
            m_XORKernel.SetupExecution(inputSize);
        }

        public override void Bind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output)
        {
            if (otherInputs == null)
                throw new ArgumentNullException("otherInputs");


            var second = otherInputs.FirstOrDefault();

            if (second == null)
                throw new ArgumentException("Nothing to bind with...");

            m_XORKernel.Run(firstInput, second, output, (int)MyJoin.MyJoinOperation.XOR, m_inputSize);

            foreach (var input in otherInputs.Skip(1)) // Exclude the second input
                m_XORKernel.Run(input, output, output, (int)MyJoin.MyJoinOperation.XOR, m_inputSize);
        }

        public override void Unbind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output)
        {
            Bind(firstInput, otherInputs, output);
        }
    }
}
