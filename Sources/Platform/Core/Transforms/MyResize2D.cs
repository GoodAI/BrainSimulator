using BrainSimulator.Task;
using BrainSimulator.Utils;
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
    [YAXSerializeAs("Resize2D")]
    public class MyResize2D : MyTransform
    {
        public MyImageScaleTask ImageScale { get; private set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0.5f)]
        public float Factor { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float FactorHeight { get; set; }    

        public override string Description { get { return "x" + Factor.ToString(); } }

        [Description("Image Scale"), MyTaskInfo(OneShot = false)]
        public class MyImageScaleTask : MyTask<MyResize2D>
        {
            private MyCudaKernel m_kernel { get; set; }

            private int inputWidth, inputHeight, outputWidth, outputHeight;

            public override void Init(Int32 nGPU)
            {
                if (Owner.Factor <= 0)
                    return;

                inputWidth = Owner.Input.ColumnHint;
                inputHeight = Owner.Input.Count / Owner.Input.ColumnHint;

                outputWidth = Owner.Output.ColumnHint;
                outputHeight = Owner.Output.Count / Owner.Output.ColumnHint;

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearResampleKernel");
                m_kernel.SetupExecution(outputWidth * outputHeight);
            }

            public override void Execute()
            {
                if (Owner.Factor <= 0)
                    return;

                m_kernel.Run(Owner.Input, Owner.Output, inputWidth, inputHeight, outputWidth, outputHeight);
            }

        }

        public override void UpdateMemoryBlocks()
        {           
            if (Input != null)
            {
                int inputWidth = Input.ColumnHint;
                int inputHeight = Input.Count / Input.ColumnHint;

                int outputWidth = (int) (inputWidth * Factor);
                int outputHeight;
                if (FactorHeight > 0)
                {
                    outputHeight = (int)(inputHeight * FactorHeight);
                }
                else
                {
                    outputHeight = (int)(inputHeight * Factor);
                }

                Output.ColumnHint = outputWidth > 0 ? outputWidth : 1;
                OutputSize = outputWidth * outputHeight;
            }
            else
            {
                OutputSize = 0;
                Output.ColumnHint = 1;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Factor > 0, this, ("Factor must be a positive decimal (currently " + Factor.ToString() + ")"));
        }
    }
}
