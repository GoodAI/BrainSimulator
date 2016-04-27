using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Transforms
{
    /// <author>GoodAI</author>
    /// <meta>mp</meta>
    /// <status>Working</status>
    /// <summary>Transforms Raw image input</summary>
    /// <description>It is meant to be used only with color, Raw 2D images.</description>
    [YAXSerializeAs("RawImageTransform")]
    public class MyRawImageTransform : MyTransform
    {
        public enum TransformTarget
        {
            RawBW,
            RGB,
            Grayscale,
            // RGBPacked // same size as raw, but channels are grouped as in RGB 
            // RGBPacked currently not used because there will be a problem with channel border alignment in pictures 
            // which are of size not divisible by 3
        }

        public MyRawImageTransformTask ImageTransform { get; private set; }

        private TransformTarget m_transformTarget = 0;

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = TransformTarget.RGB)]
        public TransformTarget Target {
            get { return m_transformTarget; }
            set
            {
                if (m_transformTarget != value)
                {
                    m_transformTarget = value;
                    switch (m_transformTarget)
                    {
                        case TransformTarget.RGB:
                            Output.Metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.RGB;
                            Output.Metadata[MemoryBlockMetadataKeys.ShowCoordinates] = false;
                            break;
                        case TransformTarget.RawBW:
                            Output.Metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.Raw;
                            Output.Metadata[MemoryBlockMetadataKeys.ShowCoordinates] = true;
                            break;
                        case TransformTarget.Grayscale:
                        default:
                            Output.Metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.RedGreenScale;
                            Output.Metadata[MemoryBlockMetadataKeys.ShowCoordinates] = false;
                            break;
                    }
                }
            }
        }

        public override string Description { get 
        {
            switch(Target)
            {
                case TransformTarget.RawBW:
                    return "Raw->Raw B&W";
                case TransformTarget.RGB:
                    return "Raw->RGB";
                case TransformTarget.Grayscale:
                    return "Raw->Grayscale";
            }
            return base.Description; 
        } }

        /// <summary>
        /// Transforms Raw image to B/W Raw, to RGB, or to Grayscale (RedGreenScale)
        /// </summary>
        [Description("Raw Image Transform"), MyTaskInfo(OneShot = false)]
        public class MyRawImageTransformTask : MyTask<MyRawImageTransform>
        {
            private MyCudaKernel m_kernel { get; set; }

            private int pixelCount;

            public override void Init(Int32 nGPU)
            {
                if (Owner.OutputSize <= 0)
                    return;

                pixelCount = Owner.InputSize;

                switch(Owner.Target)
                {
                    case TransformTarget.RGB:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "RawToRgbKernel");
                        break;
                    case TransformTarget.RawBW:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "RawToRawGrayscaleKernel");
                        break;
                    case TransformTarget.Grayscale:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Drawing\RgbaDrawing", "RawToGrayscaleKernel");
                        break;
                }

                m_kernel.SetupExecution(pixelCount);
            }

            public override void Execute()
            {
                if (Owner.OutputSize <= 0)
                    return;

                m_kernel.Run(Owner.Input, Owner.Output, pixelCount);
            }

        }

        public override void UpdateMemoryBlocks()
        {           
            if (Input != null)
            {
                TensorDimensions dims;

                switch(Target)
                {
                    case TransformTarget.RawBW:
                        dims = Input.Dims;
                        break;
                    case TransformTarget.RGB:
                        dims = Input.Dims.AddDimensions(3);
                        break;
                    case TransformTarget.Grayscale:
                    default:
                        dims = Input.Dims;
                        break;
                }

                Output.Dims = dims;
            }
            else
            {
                Output.Dims = new TensorDimensions(0);
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Input.Dims.Rank == 2, this, ("input must be a 2D image of Rank 2"));
        }
    }
}
