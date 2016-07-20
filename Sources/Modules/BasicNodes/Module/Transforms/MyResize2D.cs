using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Transforms
{
 
    /// <author>GoodAI</author>
    /// <meta>df,pd</meta>
    /// <status>Working</status>
    /// <summary>Resizes input data.</summary>
    /// <description>
    /// <h3>It is meant to be used with 2D-data (images).</h3> 
    /// <b>Exact1toN interpolation type </b> - Input image sizes (both X and Y) must be either divisible or multiples of corresponding resize factors. 
    /// In another words, to each pixel either in the output image or in the input image must correspond a rectangle of pixels  in the second image. 
    /// When increasing size, each pixel in the output image has the same value as the corresponding input image, when decreasing size, 
    /// values of each pixel in the output image are computed as average over all corresponding pixels in the input image. 
    /// </description>
    [YAXSerializeAs("Resize2D")]
    public class MyResize2D : MyTransform
    {

        public enum InterpolationType
        {
            Bilinear,
            Exact1toN
        }

        public enum ExactTransformationType
        {
            Increasing,
            Decreasing,
            Mixed
        }

        private float m_factor = 0.5f;
        public MyImageScaleTask ImageScale { get; private set; }
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0.5f)]
        public float Factor
        {
            get { return m_factor; }
            set
            {
                if (value > 0)
                {
                    m_factor = value;
                }
            }
        }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0f)]
        public float FactorHeight { get; set; }

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = InterpolationType.Bilinear)]
        public InterpolationType Interpolation { get; set; }    

        public override string Description { get { return "x" + Factor.ToString(); } }


        #region Validation

        private bool isDivisible(int a, int b)
        {
            return (a % b == 0 || b % a == 0);
        }

        public ExactTransformationType GetTransformationType()
        {
            if (outputWidth > inputWidth) 
            {
                if (outputHeight >= inputHeight) 
                {
                    return ExactTransformationType.Increasing;
                } 
                else 
                {
                    return ExactTransformationType.Mixed;
                }
            }  
            if (outputWidth < inputWidth) {
                if (outputHeight <= inputHeight) {
                    return ExactTransformationType.Decreasing;
                }
                else
                {
                    return ExactTransformationType.Mixed;
                }
            }
            //outputWidth == inputWidth
            if (outputHeight >= inputHeight) 
            {
                return ExactTransformationType.Increasing;
            } 
            else 
            {
                return ExactTransformationType.Decreasing;
            }
        }

        public bool AreParametersValid()
        {
            if (outputWidth == 0 || outputHeight == 0)
            {
                MyLog.ERROR.WriteLine(Name + ": too small factor, one output dimension would be less than 1!");
                return false;
            }
            
            if (Interpolation == InterpolationType.Exact1toN)
            {
                bool increasingWidth = outputWidth >= inputWidth;
                bool increasingHeight = outputHeight >= inputHeight;

                if (GetTransformationType() == ExactTransformationType.Mixed || !isDivisible(inputWidth, outputWidth) || !isDivisible(inputHeight, outputHeight))
                {
                   MyLog.ERROR.WriteLine("Exact1toN interpolation needs such input sizes and factors that to each pixel either in input or output image there would be exactly NxM pixels in the second image.");
                   return false;
                }   
            }
            return true;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (!AreParametersValid()) 
            {
                validator.AddError(this, "Current parameter values or their combination are not valid.");
            }
        }

        #endregion

        int inputWidth, inputHeight, outputWidth, outputHeight;

        public override void UpdateMemoryBlocks()
        {
            if (Input != null)
            {
                inputWidth = Input.ColumnHint;
                inputHeight = Input.Count / Input.ColumnHint;

                outputWidth = (int)(inputWidth * Factor);
                
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
                if (OutputSize == 0)
                {
                    OutputSize = 1;
                }
            }
            else
            {
                OutputSize = 0;
                Output.ColumnHint = 1;
            }
        }


        /// <description>Uses bilinear resampling for perfroming resize</a></description>
        [Description("Image Scale"), MyTaskInfo(OneShot = false)]
        public class MyImageScaleTask : MyTask<MyResize2D>
        {
            private MyCudaKernel m_kernel { get; set; }

            public override void Init(Int32 nGPU)
            {

                switch (Owner.Interpolation)
                {
                    case InterpolationType.Bilinear:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "BilinearResampleKernel");
                        m_kernel.SetupExecution(Owner.outputWidth * Owner.outputHeight);
                        break;
                    case InterpolationType.Exact1toN:
                        if (Owner.GetTransformationType() == ExactTransformationType.Increasing) {
                            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "ExactResampleKernel_1toN");
                        } 
                        else 
                        {
                            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\Transform2DKernels", "ExactResampleKernel_Nto1");
                        }
                        m_kernel.SetupExecution(Owner.outputWidth * Owner.outputHeight);
                        break;
                    default:
                        throw new InvalidEnumArgumentException("Unknown interpolation type " + Owner.Interpolation);
                }
            }

            public override void Execute()
            {
                if (Owner.AreParametersValid())
                {
                    m_kernel.Run(Owner.Input, Owner.Output, Owner.inputWidth, Owner.inputHeight, Owner.outputWidth, Owner.outputHeight);
                }   
            }
        }
    }
}
