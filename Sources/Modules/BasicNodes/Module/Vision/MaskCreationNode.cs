using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers; // Because of the keyboard...
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using GoodAI.Modules.Vision;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Vision
{
    /// <author>GoodAI</author>
    /// <meta>jk, jv</meta>
    /// <status>Working</status>
    /// <summary>
    /// Hides a part of the ImageInput by using some pixels from the MaskValues instead of the ImageInput.
    /// </summary>
    /// <description> 
    /// Two possibilities: choose the part of the Image to be hidden by the XCrop and YCrop values, 
    /// or probabilistic mask specified by RandomNumbersInput and MaskProbabilityInput.
    /// </description>
    public class MaskCreationNode : MyWorkingNode
    {
        //----------------------------------------------------------------------------
        // :: MEMORY BLOCKS ::
        [MyInputBlock(0)]
        public MyMemoryBlock<float> ImageInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> XCrop
        {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> YCrop
        {
            get { return GetInput(2); }
        }

        [MyInputBlock(3)]
        public MyMemoryBlock<float> MaskValuesInput
        {
            get { return GetInput(3); }
        }
        [MyInputBlock(4)]
        public MyMemoryBlock<float> RandomNumbersInput
        {
            get { return GetInput(4); }
        }
        [MyInputBlock(5)]
        public MyMemoryBlock<float> MaskProbabilityInput
        {
            get { return GetInput(5); }
        }


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> MaskOutput
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> MaskedImageOutput
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        //----------------------------------------------------------------------------
        // :: INITS  ::
        public override void UpdateMemoryBlocks()
        {
            int dim0 = (ImageInput != null && ImageInput.Dims.Rank >= 1) ? ImageInput.Dims[0] : 1;
            int dim1 = (ImageInput != null && ImageInput.Dims.Rank >= 2) ? ImageInput.Dims[1] : 1;
            int dim2 = (ImageInput != null && ImageInput.Dims.Rank >= 3) ? ImageInput.Dims[2] : 1;

            if (ImageInput.Dims.Rank < 3)
            {
                MaskOutput.Dims = new TensorDimensions(dim0, dim1);
                MaskedImageOutput.Dims = new TensorDimensions(dim0, dim1);
            }
            else
            {
                MaskOutput.Dims = new TensorDimensions(dim0, dim1, dim2);
                MaskedImageOutput.Dims = new TensorDimensions(dim0, dim1, dim2);
            }
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(ImageInput != null, this, "No input image available");
            validator.AssertError(ImageInput.Dims.Rank >= 2, this, "Input image should have rank at least 2 (2 dimensions)");

            if (!validator.ValidationSucessfull)
            {
                return;
            }

            if (Execute.Enabled)
            {
                if (MaskValuesInput == null)
                {
                    validator.AddError(this, "If the MaskByCoordinates is enabled, but no MaskValuesInput connected");
                }
                if (XCrop == null || YCrop == null)
                {
                    validator.AddError(this, "If the MaskByCoordinates is enabled, both the XCrop and YCrop have to be connected.");
                }
            }
            else if (ProbabilisticMask.Enabled)
            {
                if (MaskProbabilityInput == null)
                {
                    validator.AddError(this, "If the ProbabilisticMask is enabled, the MaskProbabilityInput has to be connected and have size 1");
                }
                if (RandomNumbersInput == null)
                {
                    validator.AddError(this, "If the ProbabilisticMask is enabled, the RandomNumbersInput with the same count as the Image Input " +
                        "has to be connected. Uniform distribution from <0,1> is expected.");
                }
            }

            if (MaskValuesInput != null && ImageInput != null && MaskValuesInput.Count != ImageInput.Count)
            {
                validator.AddError(this, "MaskValuesInput.Count != Image.Count");
            }
            if (RandomNumbersInput != null && ImageInput != null && RandomNumbersInput.Count != ImageInput.Count)
            {
                validator.AddError(this, "RandomNumbersInput.Count has to be equal to ImageInput.count");
            }
            if (MaskProbabilityInput != null && MaskProbabilityInput.Count != 1)
            {
                validator.AddError(this, "MaskProbabilityInput.Count has to be 1");
            }
        }

        [MyTaskGroup("MaskGroup")]
        public MaskCreationExecuteTask Execute { get; private set; }
        [MyTaskGroup("MaskGroup")]
        public ProbabilisticMaskCreation ProbabilisticMask { get; private set; }


        public abstract class AbstractMaskTask : MyTask<MaskCreationNode>
        {
            private MyCudaKernel m_multElementwiseKernel, maskInputKernel;

            public override void Init(int nGPU)
            {
                maskInputKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "MaskInput");
                maskInputKernel.SetupExecution(Owner.MaskOutput.Count);

                m_multElementwiseKernel = MyKernelFactory.Instance.KernelVector(Owner.GPU, KernelVector.ElementwiseMult);
                m_multElementwiseKernel.SetupExecution(Owner.MaskOutput.Count);
            }

            public override void Execute()
            {
                ProduceMask();
                ApplyMask();
            }


            /// <summary>
            /// Should produce the MaskOutput, which is then applied on the input image
            /// </summary>
            protected abstract void ProduceMask();

            /// <summary>
            /// Expects:
            /// -ImageInput: original image
            /// -MaskOutput: mask to be applied (1 means preserve the image, 0 means use the MaskValuesInput)
            /// -MaskValuesInput: what to use as a mask (the same dimension as the image), if not provided, the zeros are applied
            /// -MaskedImageOutput: produced by applying the mask to the InputImage.
            /// </summary>
            protected void ApplyMask()
            {
                if (Owner.MaskValuesInput != null)
                {
                    maskInputKernel.Run(
                        Owner.ImageInput,
                        Owner.MaskOutput,
                        Owner.MaskValuesInput,
                        Owner.MaskedImageOutput,
                        Owner.ImageInput.Count
                        );
                }
                else
                {
                    m_multElementwiseKernel.Run(
                        Owner.ImageInput,
                        Owner.MaskOutput,
                        Owner.MaskedImageOutput,
                        Owner.ImageInput.Count
                        );
                }
            }
        }

        /// <summary>
        /// Apply the MaskInput uniformly with probability specified by the MaskProbabilityInput. Probability for each pixel being masked is defined by the RandomNumbersInput.
        /// If no MaskInput specified, the zeros will be used.
        /// </summary>
        [Description("ProbabilisticMask")]
        public class ProbabilisticMaskCreation : AbstractMaskTask
        {
            private MyCudaKernel m_applyThresholdKernel;

            public override void Init(int nGPU)
            {
                base.Init(nGPU);

                m_applyThresholdKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "ApplyThreshold");
                m_applyThresholdKernel.SetupExecution(Owner.MaskOutput.Count);
            }

            protected override void ProduceMask()
            {
                if (Owner.MaskProbabilityInput == null || Owner.RandomNumbersInput == null)
                {
                    MyLog.WARNING.WriteLine("ProbabilisticMask enabled, but either no MaskProbabilityInput or RandomNumbersInput not connected, not computing");
                    return;
                }

                m_applyThresholdKernel.Run(
                    Owner.RandomNumbersInput,
                    Owner.MaskOutput,
                    Owner.MaskProbabilityInput,
                    Owner.MaskOutput.Count
                    );
            }
        }

        /// <summary>
        /// Apply the MaskInput by the part of the image defined by the XCrop and YCrop inputs. If no MaskInput specified, the zeros will be used.
        /// </summary>
        [Description("MaskByCoordinates")]
        public class MaskCreationExecuteTask : AbstractMaskTask
        {
            private MyCudaKernel kerX, kerY;

            public override void Init(int nGPU)
            {
                base.Init(nGPU);

                kerX = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "SetMatrixVauleMinMaxX");
                kerX.SetupExecution(Owner.MaskOutput.Count);

                kerY = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "SetMatrixVauleMinMaxY");
                kerY.SetupExecution(Owner.MaskOutput.Count);
            }

            private bool CropHasUsefullValueAndCopy2Host(MyMemoryBlock<float> Crop)
            {
                if (Crop == null)
                {
                    MyLog.WARNING.WriteLine("Crop named " + Crop.Name + " not connected, not cropping this dimension");
                    return false;
                }
                Crop.SafeCopyToHost();
                // deadband aroud zero
                if (Crop.Host[0] < 0.1f && Crop.Host[0] > -0.1f)
                    return false;
                return true;
            }

            protected override void ProduceMask()
            {
                Owner.MaskOutput.Fill(1.0f);

                if (CropHasUsefullValueAndCopy2Host(Owner.XCrop))
                {
                    if (Owner.XCrop.Host[0] > 0f)
                    {
                        kerX.Run(Owner.MaskOutput, Owner.MaskOutput.Dims[0], Owner.MaskOutput.Count, 0, (int)(Owner.XCrop.Host[0] * Owner.MaskOutput.Dims[0]), 0f);
                    }
                    else
                    {
                        kerX.Run(Owner.MaskOutput, Owner.MaskOutput.Dims[0], Owner.MaskOutput.Count, (int)Owner.MaskOutput.Dims[0] + (int)(Owner.XCrop.Host[0] * Owner.MaskOutput.Dims[0]), (int)Owner.MaskOutput.Dims[0], 0f);
                    }
                }
                if (CropHasUsefullValueAndCopy2Host(Owner.YCrop))
                {
                    if (Owner.YCrop.Host[0] > 0f)
                    {
                        kerY.Run(Owner.MaskOutput, Owner.MaskOutput.Dims[0], Owner.MaskOutput.Count, 0, (int)(Owner.YCrop.Host[0] * Owner.MaskOutput.Dims[1]), 0f);
                    }
                    else
                    {
                        kerY.Run(Owner.MaskOutput, Owner.MaskOutput.Dims[0], Owner.MaskOutput.Count, (int)Owner.MaskOutput.Dims[1] + (int)(Owner.YCrop.Host[0] * Owner.MaskOutput.Dims[1]), (int)Owner.MaskOutput.Dims[1], 0f);
                    }
                }
            }
        }
    }
}