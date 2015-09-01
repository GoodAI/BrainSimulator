using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda.CudaFFT;
using ManagedCuda.VectorTypes;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Learning transformations of Symbolic Pointers with gradient method</summary>
    /// <description>Works (probably) only on linear transformations</description>
    class MyTransformLearningNode : MyWorkingNode
    {

        [MyInputBlock(0), Description("Input")]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1), Description("Goal")]
        public MyMemoryBlock<float> Goal
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(0), Description("Transform")]
        public MyMemoryBlock<float> Transform
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public MyMemoryBlock<cuFloatComplex> FirstInputFFT { get; private set; }
        public MyMemoryBlock<cuFloatComplex> SecondInputFFT { get; private set; }
        public MyMemoryBlock<float> Temp { get; private set; }
        public MyMemoryBlock<float> Difference { get; private set; }

        public int InputSize
        {
            get { return Input != null ? Input.Count : 0; }
        }

        public override void UpdateMemoryBlocks()
        {
            FirstInputFFT.Count = InputSize + 1;
            SecondInputFFT.Count = InputSize + 1;

            FirstInputFFT.ColumnHint = Input != null ? Input.ColumnHint : 1;
            SecondInputFFT.ColumnHint = Goal != null ? Goal.ColumnHint : 1;

            Temp.Count = InputSize;
            Temp.ColumnHint = Input != null ? Input.ColumnHint : 1;

            Difference.Count = InputSize;
            Difference.ColumnHint = Input != null ? Input.ColumnHint : 1;

            Transform.Count = InputSize;
            Transform.ColumnHint = Input != null ? Input.ColumnHint : 1;
        }

        public override string Description
        {
            get
            {
                return "T: Input ? T = Goal";
            }
        }

        public MyLearningTask BindInputs { get; private set; }

        [Description("Learn Transform")]
        public class MyLearningTask : MyTask<MyTransformLearningNode>
        {     
            private CudaFFTPlan1D fft;
            private CudaFFTPlan1D ifft;
            private MyCudaKernel m_kernel;
            private MyCudaKernel m_involutionKernel;
            private MyCudaKernel m_linearCombKernel;
            private MyCudaKernel m_normalKernel;

            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = 0.25f)]
            public float StepSize { get; set; }

            public override void Init(int nGPU)
            {
                fft = new CudaFFTPlan1D(Owner.InputSize, cufftType.R2C, 1);
                ifft = new CudaFFTPlan1D(Owner.InputSize, cufftType.C2R, 1);

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "MulComplexElementWise");
                m_kernel.SetupExecution(Owner.InputSize + 1);

                m_involutionKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "InvolveVector");
                m_involutionKernel.SetupExecution(Owner.InputSize - 1);

                m_linearCombKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "LinearCombinationKernel");
                m_linearCombKernel.SetupExecution(Owner.InputSize);

                m_normalKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
                m_normalKernel.SetupExecution(Owner.InputSize);
            }


            public void Binding(MyMemoryBlock<float> first, MyMemoryBlock<float> second, MyMemoryBlock<float> temp, MyMemoryBlock<float> destination, bool DoQuery)
            {

                fft.Exec(first.GetDevicePtr(Owner), Owner.FirstInputFFT.GetDevicePtr(Owner));

                if (DoQuery)
                {
                    m_involutionKernel.Run(second, temp, second.Count);
                    fft.Exec(temp.GetDevicePtr(Owner), Owner.SecondInputFFT.GetDevicePtr(Owner));
                }
                else
                {
                    fft.Exec(second.GetDevicePtr(Owner), Owner.SecondInputFFT.GetDevicePtr(Owner));
                }

                m_kernel.Run(Owner.FirstInputFFT, Owner.SecondInputFFT, Owner.SecondInputFFT, Owner.InputSize + 1);

                ifft.Exec(Owner.SecondInputFFT.GetDevicePtr(Owner), temp.GetDevicePtr(Owner));

                float factor = 1.0f / Owner.Transform.Count;
                if (factor != 1)
                {
                    m_normalKernel.Run(0, 0, factor, 0, temp, destination, Owner.InputSize);
                }

            }

            public override void Execute()
            {
                // bind transform and first input -> temp
                Binding(Owner.Input, Owner.Transform, Owner.Temp, Owner.Difference, false);

                // substract transformed vector (temp) from goal (Owner.SecondInput) -> temp
                m_linearCombKernel.Run(Owner.Goal, 1.0f, 0, Owner.Difference, -1.0f, 0, Owner.Difference, 0, Owner.InputSize);

                // unbind temp from first input -> temp
                Binding(Owner.Difference, Owner.Input, Owner.Temp, Owner.Difference, true);

                // scale temp (which is gradient) by step size and add to Owner.Output, which is the learned transform
                m_linearCombKernel.Run(Owner.Transform, 1.0f, 0, Owner.Difference, StepSize, 0, Owner.Transform, 0, Owner.InputSize);
            }
        }
    }
}
