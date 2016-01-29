using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Transforms
{

    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>2D image analyzes.</summary>
    /// <description>
    ///    Given the input image (2D matrix as memory block), the node is able to apply image processing algorithm. So far, only the optical flow by Lucas and Kanade (1981) is available.
    ///    
    /// </description>
    [YAXSerializeAs("Analyze2D")]
    public class MyAnalyze2D : MyTransform
    {        
        public MyOpticalFlowTask OpticalFlow { get; private set; }

        public MyMemoryBlock<float> Temp { get; private set; }
        public MyMemoryBlock<float> Derivatives { get; private set; }
        public MyMemoryBlock<float> LastInput { get; private set; }

        public MyMemoryBlock<float> GlobalFlow { get; private set; }      

        public override void UpdateMemoryBlocks()
        {
            OutputSize = InputSize * 2;
            LastInput.Count = InputSize;
            Temp.Count = InputSize;
            Derivatives.Count = InputSize * 5;
            GlobalFlow.Count = 2;

            if (Input != null)
            {
                Output.ColumnHint = Input.ColumnHint;                
                Temp.ColumnHint = Input.ColumnHint;                
                Derivatives.ColumnHint = Input.ColumnHint;
            }                     
        }
        /// <summary>
        ///   Optical flow algorithm tracks all  pixels (or specific features) in an image all the time. The is based on the following observation:
        ///   If there is an object in the image, the difference between corresponding parts (pixels) of the object in two near-by frames should be constant.
        ///   This formulation leads to the set of equations and their solution is solution of the optical flow: <b>the movement of image pixels/features</b>.
        /// </summary>
        [Description("Optical Flow (Lucas-Kanade)")]
        public class MyOpticalFlowTask : MyTask<MyAnalyze2D>
        {
            [YAXSerializableField]
            [MyBrowsable, Category("Params")]
            public bool SubtractGlobalFlow { get; set; }

            public static float[] BLUR_KERNEL = { 0.077847f, 0.123317f, 0.077847f, 0.123317f, 0.195346f, 0.123317f, 0.077847f, 0.123317f, 0.077847f };            
            public static float[] SOBEL_KERNEL_X = { 0, 0, 0, -1, 0, 1, 0, 0, 0 };
            public static float[] SOBEL_KERNEL_Y = { 0, -1, 0, 0, 0, 0, 0, 1, 0 };

            private int imageWidth, imageHeight;

            private MyCudaKernel m_kernel;
            private MyCudaKernel m_derivativeKernel;
            private MyCudaKernel m_velocityKernel;
            private MyCudaKernel m_finalizeKernel;
            private MyReductionKernel<float> m_reductionKernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\ConvolutionSingle", "Convolution3x3Single");                
                m_derivativeKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\OpticalFlow", "PrepareDerivativesKernel");
                m_velocityKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\OpticalFlow", "EvaluateVelocityKernel");
                m_finalizeKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\OpticalFlow", "FinalizeVelocityKernel");
                m_reductionKernel = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_Sum_f);

                imageWidth = Owner.Input.ColumnHint;
                imageHeight = Owner.InputSize / imageWidth;

                m_kernel.SetupExecution(Owner.InputSize);
                m_derivativeKernel.SetupExecution(Owner.InputSize);
                m_velocityKernel.SetupExecution(Owner.InputSize);
                m_finalizeKernel.SetupExecution(Owner.InputSize);
            }

            public override void Execute()
            {
                Owner.Output.Fill(0);
                Owner.Temp.CopyFromMemoryBlock(Owner.Input, 0, 0, Owner.InputSize);

                CudaDeviceVariable<float> derivVar = Owner.Derivatives.GetDevice(Owner);

                m_kernel.SetupExecution(Owner.InputSize);
                m_kernel.SetConstantVariable("D_KERNEL", BLUR_KERNEL);
                m_kernel.Run(Owner.Input, Owner.Temp, imageWidth, imageHeight);

                if (SimulationStep == 0)
                {
                    Owner.LastInput.CopyFromMemoryBlock(Owner.Temp, 0, 0, Owner.InputSize);
                }

                m_kernel.SetConstantVariable("D_KERNEL", SOBEL_KERNEL_X);
                m_kernel.Run(Owner.Temp, Owner.Derivatives, imageWidth, imageHeight);
                m_kernel.SetConstantVariable("D_KERNEL", SOBEL_KERNEL_Y);
                m_kernel.Run(Owner.Temp, derivVar.DevicePointer + Owner.InputSize * derivVar.TypeSize, imageWidth, imageHeight);

                m_derivativeKernel.Run(Owner.Temp, Owner.LastInput, Owner.Derivatives, imageWidth, imageHeight);

                m_kernel.SetupExecution(Owner.Derivatives.Count);
                m_kernel.SetConstantVariable("D_KERNEL", BLUR_KERNEL);

                for (int i = 0; i < 5; i++)
                {
                    CUdeviceptr ptr = derivVar.DevicePointer + i * Owner.InputSize * derivVar.TypeSize;

                    m_kernel.Run(ptr, ptr, imageWidth, imageHeight);
                    m_kernel.Run(ptr, ptr, imageWidth, imageHeight);
                    m_kernel.Run(ptr, ptr, imageWidth, imageHeight);
                }

                m_velocityKernel.Run(Owner.Derivatives, Owner.Output, imageWidth, imageHeight);

                if (SubtractGlobalFlow)
                {
                    //ZXC m_reductionKernel.Run(Owner.Output, Owner.GlobalFlow, Owner.InputSize, 0, 0, 1,0);
                    m_reductionKernel.outOffset = 0;
                    m_reductionKernel.inOffset = 0;
                    m_reductionKernel.Run(Owner.Output, Owner.GlobalFlow);
                    //ZXC m_reductionKernel.Run(Owner.Output, Owner.GlobalFlow, Owner.InputSize, Owner.InputSize, 1, 1,0);
                    m_reductionKernel.outOffset = Owner.InputSize;
                    m_reductionKernel.inOffset = 1;
                    m_reductionKernel.Run(Owner.Output, Owner.GlobalFlow);

                    Owner.GlobalFlow.SafeCopyToHost();

                    float l = (float)Math.Sqrt(Owner.GlobalFlow.Host[0] * Owner.GlobalFlow.Host[0] + Owner.GlobalFlow.Host[1] * Owner.GlobalFlow.Host[1]);

                    if (l > 0)
                    {
                        Owner.GlobalFlow.Host[0] /= l;
                        Owner.GlobalFlow.Host[1] /= l;
                    }

                    Owner.GlobalFlow.SafeCopyToDevice();                    
                }
            }
        }       

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertWarning(Input.ColumnHint != 1, this, "Node is attached to non-matrix input");
        }                
    }
}
