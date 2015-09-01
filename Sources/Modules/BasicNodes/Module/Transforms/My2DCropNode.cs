using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Transforms
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Crops a 2D image</summary>
    /// <description></description>
    public class My2DCropNode : MyTransform
    {
        public MyImageCropTask ImageCrop { get; private set; }

        [MyBrowsable, Category("Crop"), Description("Negative to crop, positive to add a blank margin")]
        [YAXSerializableField(DefaultValue = 0)]
        public int LeftMargin { get; set; }

        [MyBrowsable, Category("Crop"), Description("Negative to crop, positive to add a blank margin")]
        [YAXSerializableField(DefaultValue = 0)]
        public int RightMargin { get; set; }

        [MyBrowsable, Category("Crop"), Description("Negative to crop, positive to add a blank margin")]
        [YAXSerializableField(DefaultValue = 0)]
        public int TopMargin { get; set; }

        [MyBrowsable, Category("Crop"), Description("Negative to crop, positive to add a blank margin")]
        [YAXSerializableField(DefaultValue = 0)]
        public int BottomMargin { get; set; }


        public override string Description { get { return LeftMargin.ToString("+#;-#;0") + " " + TopMargin.ToString("+#;-#;0") + " " + RightMargin.ToString("+#;-#;0") + " " + BottomMargin.ToString("+#;-#;0"); } }

        /// <description>Performs cropping.</description>
        [Description("Crop"), MyTaskInfo(OneShot = false)]
        public class MyImageCropTask : MyTask<My2DCropNode>
        {
            private MyCudaKernel m_kernel;

            [MyBrowsable, Category("Crop"), Description("Value to fill the blanks")]
            [YAXSerializableField(DefaultValue = 0)]
            public float FillValue { get; set; }


            int inputWidth;
            int inputHeight;
            int outputWidth;
            int outputHeight;

            public override void Init(Int32 nGPU)
            {
                inputWidth = Owner.Input.ColumnHint;
                inputHeight = Owner.Input.Count / inputWidth;

                outputWidth = Owner.Output.ColumnHint;
                outputHeight = Owner.Output.Count / outputWidth;

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "Crop2DKernel");
            }

            public override void Execute()
            {
                m_kernel.SetupExecution(outputWidth * outputHeight);
                m_kernel.Run(Owner.Input, Owner.Output, inputWidth, inputHeight, outputWidth, outputWidth * outputHeight, Owner.LeftMargin, Owner.TopMargin, FillValue);
            }

        }

        public override void UpdateMemoryBlocks()
        {           
            if (Input != null)
            {
                int inputWidth = Input.ColumnHint;
                int inputHeight = Input.Count / Input.ColumnHint;

                int outputWidth = inputWidth + (int)LeftMargin + (int)RightMargin;
                int outputHeight = inputHeight + (int)TopMargin + (int)BottomMargin;

                if (outputWidth > 0 && outputHeight > 0)
                {
                    OutputSize = outputWidth * outputHeight;
                    Output.ColumnHint = outputWidth;
                }
                else
                {
                    OutputSize = 0;
                    Output.ColumnHint = 1;
                }
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

            if (Input != null)
            {
                int inputWidth = Input.ColumnHint;
                int inputHeight = Input.Count / inputWidth;

                validator.AssertError(inputWidth + LeftMargin + RightMargin > 0, this, "Left or right margin is too big");
                validator.AssertError(inputHeight + TopMargin + BottomMargin > 0, this, "Top or Bottom margin is too big");
            }
        }
    }
}
