using BrainSimulator.Memory;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.Transforms
{
    [YAXSerializeAs("Filter2D")]
    public class MyFilter2D : MyTransform
    {        
        [MyTaskGroup("Filter")]
        public MyVariance3x3Task Variance3x3 { get; private set; }
        [MyTaskGroup("Filter")]
        public MyEdgeDetectionTask EdgeDetection { get; private set; }
        [MyTaskGroup("Filter")]
        public MyGaussianBlurTask GaussianBlur { get; private set; }
        [MyTaskGroup("Filter")]
        public MySobelEdgeTask SobelEdge { get; private set; }        

        public MyMemoryBlock<float> Temp { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            Temp.Count = Output.Count;
            Temp.ColumnHint = Output.ColumnHint;
        }

        [Description("Variance 3x3")]
        public class MyVariance3x3Task : MyTask<MyFilter2D>
        {
            private MyCudaKernel m_kernel { get; set; }

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Variance3x3Kernel");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.InputSize);
                m_kernel.Run(Owner.Input, Owner.Output, Owner.InputSize, Owner.Input.ColumnHint);
            }
        }

        [Description("Edge detection")]
        public class MyEdgeDetectionTask : MyTask<MyFilter2D>
        {
            private MyCudaKernel m_kernel;

            public override void Init(Int32 nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\EdgeDetectionKernel");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(Owner.InputSize);
                m_kernel.Run(Owner.Input, Owner.Output, Owner.InputSize, Owner.Input.ColumnHint);
            }
        }

        [Description("Gaussian Blur (3x3)")]
        public class MyGaussianBlurTask : MyTask<MyFilter2D>
        {
            private MyCudaKernel m_kernel;
            public static float[] KERNEL = { 0.077847f, 0.123317f, 0.077847f, 0.123317f, 0.195346f, 0.123317f, 0.077847f, 0.123317f, 0.077847f };
            int imageWidth, imageHeight;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\ConvolutionSingle", "Convolution3x3Single");                
                
                imageWidth = Owner.Input.ColumnHint;
                imageHeight = Owner.InputSize / imageWidth;

                m_kernel.SetupExecution(Owner.Input.Count);                
            }

            public override void Execute()
            {
                Owner.Output.Fill(0);

                m_kernel.SetConstantVariable("D_KERNEL", KERNEL);
                m_kernel.Run(Owner.Input, Owner.Output, imageWidth, imageHeight);                
            }
        }

        [Description("Sobel Edge Detection (3x3)")]
        public class MySobelEdgeTask : MyTask<MyFilter2D>
        {
            public static float[] KERNEL_X = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
            public static float[] KERNEL_Y = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

            private int imageWidth, imageHeight;
            private MyCudaKernel m_kernel { get; set; }
            private MyCudaKernel m_finalizeKernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\ConvolutionSingle", "Convolution3x3Single");
                m_finalizeKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "LengthFromElements");                

                imageWidth = Owner.Input.ColumnHint;
                imageHeight = Owner.InputSize / imageWidth;

                m_kernel.SetupExecution(Owner.Input.Count);
                m_finalizeKernel.SetupExecution(Owner.Input.Count);
            }

            public override void Execute()
            {
                Owner.Output.Fill(0);

                m_kernel.SetConstantVariable("D_KERNEL", KERNEL_X);
                m_kernel.Run(Owner.Input, Owner.Output, imageWidth, imageHeight);

                m_kernel.SetConstantVariable("D_KERNEL", KERNEL_Y);
                m_kernel.Run(Owner.Input, Owner.Temp, imageWidth, imageHeight);

                m_finalizeKernel.Run(Owner.Output, Owner.Temp, Owner.Output, Owner.Output.Count);
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertWarning(Input.ColumnHint != 1, this, "Node is attached to non-matrix input");
        }                
    }
}
