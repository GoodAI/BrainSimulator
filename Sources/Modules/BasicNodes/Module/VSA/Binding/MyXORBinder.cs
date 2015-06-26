
using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Transforms;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrainSimulator.VSA
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


        public override void Bind(CUdeviceptr firstInput, params CUdeviceptr[] otherInputs)
        {
            if (otherInputs == null)
                otherInputs = new CUdeviceptr[] { firstInput };

            var output = otherInputs[otherInputs.Length - 1];
            m_XORKernel.Run(firstInput, otherInputs[0], output, (int)MyJoin.MyJoinOperation.XOR, m_inputSize);

            for (int i = 1; i < otherInputs.Length - 1; ++i)
                m_XORKernel.Run(otherInputs[i], output, output, (int)MyJoin.MyJoinOperation.XOR, m_inputSize);
        }

        public override void Unbind(CUdeviceptr firstInput, params CUdeviceptr[] otherInputs)
        {
            Bind(firstInput, otherInputs);
        }
    }
}
