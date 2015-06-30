using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Tasks;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;

namespace CustomModels.NeuralNetwork.Tasks
{
    class MyConvolutionTasks
    {
    }

    [Description("PoolingForward"), MyTaskInfo(OneShot = false)]
    public class MyPoolingForwardTask : MyAbstractForwardTask<MyPoolingLayer>
    {
        private MyCudaKernel m_kernel;

        public MyPoolingForwardTask() { } //parameterless constructor

        public override void Init(int nGPU) //Kernel initialization
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\PoolingKernel", "PoolingForwardKernel");
        }

        public override void Execute() //Task execution
        {
            m_kernel.SetupExecution(Owner.Neurons);
            m_kernel.Run(
                Owner.Input,
                Owner.Output,
                Owner.ActivatedNeurons,
                Owner.InputWidth, Owner.InputWidth * Owner.InputHeight,
                Owner.FilterWidth, Owner.FilterHeight,
                Owner.HorizontalStride, Owner.VerticalStride,
                Owner.OutputWidth * Owner.OutputHeight,
                Owner.Neurons
                );
            MyLog.DEBUG.WriteLine("Pooling.");
        }
    }

    [Description("PoolingBackward"), MyTaskInfo(OneShot = false)]
    public class MyPoolingBackwardTask : MyAbstractBackDeltaTask<MyPoolingLayer>
    {
        private MyCudaKernel m_kernel;

        public MyPoolingBackwardTask() { } //parameterless constructor

        public override void Init(int nGPU) //Kernel initialization
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\PoolingKernel", "PoolingBackwardKernel");
        }

        public override void Execute() //Task execution
        {
            if (Owner.ParentNetwork.SGD.Enabled) // SGD
            {
                m_kernel.SetupExecution(Owner.Neurons);
                m_kernel.Run(
                    Owner.Input,
                    Owner.ParentNetwork.SGD.TrainingRate,
                    Owner.ParentNetwork.SGD.Momentum,
                    Owner.Input.Count,
                    Owner.Neurons
                    );
            }
            else if (Owner.ParentNetwork.RMS.Enabled) // RMSProp
            {
                // TODO: Implement RProp!
                MyLog.ERROR.WriteLine("No RMSProp not yet implemented for fully convolution layers");
            }
            else
                MyLog.ERROR.WriteLine("No backprop task selected in " + Owner.ParentNetwork + " please select SGD or RProp to perform backpropagation");
        }
    }

    [Description("PadImage"), MyTaskInfo(OneShot = false)]
    public class MyPadImageTask : MyTask<MyConvolutionLayer>
    {
        private MyCudaKernel m_kernel;

        public MyPadImageTask() { } //parameterless constructor

        public override void Init(int nGPU) //Kernel initialization
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "PadImageKernel");
        }

        public override void Execute() //Task execution
        {
            if (Owner.ZeroPadding <= 0) return;

            Owner.PaddedImage.Fill(0);

            m_kernel.SetupExecution(Owner.Input.Count);
            m_kernel.Run(
                Owner.Input,
                Owner.PaddedImage,
                Owner.InputWidth,
                Owner.ZeroPadding,
                Owner.InputWidth*Owner.InputHeight,
                (Owner.InputWidth + Owner.ZeroPadding + Owner.ZeroPadding) * (Owner.InputHeight + Owner.ZeroPadding + Owner.ZeroPadding),
                Owner.Input.Count
                );
        }
    }

    [Description("ConvolutionForward"), MyTaskInfo(OneShot = false)]
    public class MyConvolutionForwardTask : MyAbstractForwardTask<MyConvolutionLayer>
    {
        private MyCudaKernel m_kernel;

        public MyConvolutionForwardTask() { } //parameterless constructor

        public override void Init(int nGPU) //Kernel initialization
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "ConvolutionForwardKernel");
        }

        public override void Execute() //Task execution
        {
            m_kernel.SetupExecution(Owner.Output.Count);

            // use the input image as it is
            if (Owner.ZeroPadding <= 0)

                m_kernel.Run(
                    Owner.Input,
                    Owner.Weights,
                    Owner.Bias,
                    Owner.Output,
                    Owner.FilterWidth, Owner.FilterHeight,
                    Owner.InputDepth,
                    Owner.FilterWidth * Owner.FilterHeight,
                    Owner.FilterWidth * Owner.FilterHeight * Owner.InputDepth,
                    Owner.InputWidth * Owner.InputHeight,
                    Owner.InputWidth,
                    Owner.OutputWidth * Owner.OutputHeight,
                    1 + (Owner.InputWidth - Owner.FilterWidth) / Owner.HorizontalStride, //1 + (inputWidth - filterWidth) / horStride
                    Owner.HorizontalStride, Owner.VerticalStride,
                    Owner.Output.Count
                );
            // do and use zero padding
            else
                m_kernel.Run(
                    Owner.PaddedImage,
                    Owner.Weights,
                    Owner.Bias,
                    Owner.Output,
                    Owner.FilterWidth, Owner.FilterHeight,
                    Owner.InputDepth,
                    Owner.FilterWidth * Owner.FilterHeight,
                    Owner.FilterWidth * Owner.FilterHeight * Owner.InputDepth,
                    (Owner.InputWidth + Owner.ZeroPadding + Owner.ZeroPadding) * (Owner.InputHeight + Owner.ZeroPadding + Owner.ZeroPadding),
                    (Owner.InputWidth + Owner.ZeroPadding + Owner.ZeroPadding),
                    Owner.OutputWidth * Owner.OutputHeight,
                    1 + ((Owner.InputWidth + Owner.ZeroPadding + Owner.ZeroPadding) - Owner.FilterWidth) / Owner.HorizontalStride, //1 + (inputWidth - filterWidth) / horStride
                    Owner.HorizontalStride, Owner.VerticalStride,
                    Owner.Output.Count
                );

        }
    }

    public class MyConvolutionBackwardTask : MyAbstractBackDeltaTask<MyConvolutionLayer>
    {
        public override void Init(int nGPU)
        {
            MyLog.DEBUG.WriteLine("Convolution Back Init");
        }

        public override void Execute()
        {
            MyLog.DEBUG.WriteLine("Convolution Back Execute");
        }
    }


    [Description("InitLayer"), MyTaskInfo(OneShot = true)]
    public class MyConvolutionInitLayerTask : MyTask<MyConvolutionLayer>
    {
        public MyConvolutionInitLayerTask() { } //parameterless constructor
        public override void Init(int nGPU) { } //Kernel initialization

        public override void Execute() //Task execution
        {
            // init vars to 0
            Owner.PreviousBiasDelta.Fill(0f);
            Owner.PreviousWeightDelta.Fill(0f);

            // init random weights
            // float stdDev = 1.0f / (float)Math.Sqrt(Owner.Input.Count + 1);
            float stdDev = 0.01f;
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Weights.GetDevice(Owner), 0, stdDev);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Bias.GetDevice(Owner), 0, stdDev);
//            Owner.Weights.Fill(1f);
//            Owner.Bias.Fill(0f);
        }
    }
}
