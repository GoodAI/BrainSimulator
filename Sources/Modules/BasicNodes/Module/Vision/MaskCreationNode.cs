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
    /// <meta>jk</meta>
    /// <status>Working</status>
    /// <summary>
    /// ?
    /// </summary>
    /// <description> ? </description>
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
            //base.Validate(validator); /// base checking 
            validator.AssertError(ImageInput != null, this, "No input image available");
            validator.AssertError(ImageInput.Dims.Rank >= 2, this, "Input image should have rank at least 2 (2 dimensions)");

            if (MaskValuesInput != null && ImageInput != null && MaskValuesInput.Count != ImageInput.Count)
            {
                validator.AddError(this, "MaskValuesInput.Count != Image.Count");
            }
        }

        public MaskCreationExecuteTask Execute { get; private set; }

        [Description("Execute")]
        public class MaskCreationExecuteTask : MyTask<MaskCreationNode>
        {
            MyCudaKernel kerX, kerY, maskInputKernel;
            MyCudaKernel m_multElementwiseKernel;

            public override void Init(int nGPU)
            {
                kerX = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "SetMatrixVauleMinMaxX");
                kerX.SetupExecution(Owner.MaskOutput.Count);

                kerY = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "SetMatrixVauleMinMaxY");
                kerY.SetupExecution(Owner.MaskOutput.Count);

                maskInputKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "MaskInput");
                maskInputKernel.SetupExecution(Owner.MaskOutput.Count);

                m_multElementwiseKernel = MyKernelFactory.Instance.KernelVector(Owner.GPU, KernelVector.ElementwiseMult);
                m_multElementwiseKernel.SetupExecution(Owner.MaskOutput.Count);
            }

            private bool CropHasUsefullValueAndCopy2Host(MyMemoryBlock<float> Crop)
            {
                if (Crop == null)
                    return false;
                Crop.SafeCopyToHost();
                // deadband aroud zero
                if (Crop.Host[0] < 0.1f && Crop.Host[0] > -0.1f)
                    return false;
                return true;
            }

            public override void Execute()
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
    }
}