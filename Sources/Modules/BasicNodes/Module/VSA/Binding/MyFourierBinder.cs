using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Modules.Transforms;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaFFT;
using System;
using System.Collections.Generic;
using ManagedCuda;

namespace GoodAI.Modules.VSA
{
    public class MyFourierBinder : MySymbolBinderBase
    {
        private CudaFFTPlan1D m_fft;
        private CudaFFTPlan1D m_ifft;

        private MyCudaKernel m_mulkernel;
        private MyCudaKernel m_involutionKernel;
        private MyCudaKernel m_inversionKernel;
        private MyCudaKernel m_normalKernel;

        private MyProductKernel<float> m_dotKernel;

        private int m_firstFFTOffset;
        private int m_secondFFTOffset;
        private int m_tempOffset;

        CudaStream m_stream;


        public MyFourierBinder(MyWorkingNode owner, int inputSize, MyMemoryBlock<float> tempBlock)
            : base(owner, inputSize, tempBlock)
        {
            m_stream = new CudaStream();

            m_fft = new CudaFFTPlan1D(inputSize, cufftType.R2C, 1);
            m_fft.SetStream(m_stream.Stream);
            m_ifft = new CudaFFTPlan1D(inputSize, cufftType.C2R, 1);
            m_ifft.SetStream(m_stream.Stream);

            m_mulkernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Common\CombineVectorsKernel", "MulComplexElementWise");
            m_mulkernel.SetupExecution(inputSize + 1);

            m_involutionKernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Common\CombineVectorsKernel", "InvolveVector");
            m_involutionKernel.SetupExecution(inputSize - 1);

            m_inversionKernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Transforms\InvertValuesKernel", "InvertLengthComplexKernel");
            m_inversionKernel.SetupExecution(inputSize);

            m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(owner, owner.GPU, ProductMode.f_DotProduct_f);

            m_normalKernel = MyKernelFactory.Instance.Kernel(owner.GPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
            m_normalKernel.SetupExecution(inputSize);

            m_firstFFTOffset = 0;
            m_secondFFTOffset = (inputSize + 1) * 2;
            m_tempOffset = (inputSize + 1) * 4;

            Denominator = inputSize;
        }


        public static int GetTempBlockSize(int inputSize)
        {
            return inputSize * 5 + 4;
        }


        public override void Bind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output)
        {
            m_fft.Exec(firstInput, m_tempBlock.GetDevicePtr(m_owner, m_secondFFTOffset));

            foreach (var input in otherInputs)
            {
                m_fft.Exec(input, m_tempBlock.GetDevicePtr(m_owner, m_firstFFTOffset));
                m_mulkernel.RunAsync(
                    m_stream,
                    m_tempBlock.GetDevicePtr(m_owner, m_firstFFTOffset),
                    m_tempBlock.GetDevicePtr(m_owner, m_secondFFTOffset),
                    m_tempBlock.GetDevicePtr(m_owner, m_secondFFTOffset), m_inputSize + 1);
            }

            FinishBinding(output);
        }

        public override void Unbind(CUdeviceptr firstInput, IEnumerable<CUdeviceptr> otherInputs, CUdeviceptr output)
        {
            m_fft.Exec(firstInput, m_tempBlock.GetDevicePtr(m_owner, m_secondFFTOffset));

            foreach (var input in otherInputs)
            {
                m_involutionKernel.RunAsync(m_stream, input, m_tempBlock.GetDevicePtr(m_owner, m_tempOffset), m_inputSize);
                m_fft.Exec(m_tempBlock.GetDevicePtr(m_owner, m_tempOffset), m_tempBlock.GetDevicePtr(m_owner, m_firstFFTOffset));

                if (ExactQuery)
                    m_inversionKernel.RunAsync(m_stream, m_tempBlock.GetDevicePtr(m_owner, m_firstFFTOffset), m_tempBlock.GetDevicePtr(m_owner, m_firstFFTOffset), m_inputSize);

                m_mulkernel.RunAsync(
                    m_stream,
                    m_tempBlock.GetDevicePtr(m_owner, m_secondFFTOffset),
                    m_tempBlock.GetDevicePtr(m_owner, m_firstFFTOffset),
                    m_tempBlock.GetDevicePtr(m_owner, m_secondFFTOffset), m_inputSize + 1);
            }

            FinishBinding(output);
        }

        private void FinishBinding(CUdeviceptr output)
        {
            m_ifft.Exec(m_tempBlock.GetDevicePtr(m_owner, m_secondFFTOffset), output);

            float factor = 1.0f;

            if (NormalizeOutput)
            {
                //ZXC m_dotKernel.RunAsync(m_stream, m_tempBlock, 0, output, output, m_inputSize, /* distributed: */ 0);
                m_dotKernel.RunAsync(m_stream, m_tempBlock, output, output, m_inputSize);
                m_tempBlock.SafeCopyToHost(0, 1);

                if (m_tempBlock.Host[0] > 0.000001f)
                    factor /= (float)(Math.Sqrt(m_tempBlock.Host[0]));
            }
            else
            {
                factor = 1.0f / Denominator;
            }

            if (factor != 1.0f)
            {
                m_normalKernel.Run(0, 0, factor, 0, output, output, m_inputSize);
            }
            else
            {
                m_stream.Synchronize();
            }
        }
    }
}
