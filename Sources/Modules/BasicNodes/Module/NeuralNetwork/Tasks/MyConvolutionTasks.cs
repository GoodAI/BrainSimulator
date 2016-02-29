using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Tasks;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;

namespace CustomModels.NeuralNetwork.Tasks
{

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// Performs MAX pooling forward pass. Chooses the max value from each receptive field and its each position (determined by FilterW/H and Stride parameters).
    /// </summary>
    /// <description></description>
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
                Owner.OutputWidth, Owner.OutputWidth * Owner.OutputHeight,
                Owner.Neurons
            );
            MyLog.DEBUG.WriteLine("Pooling.");
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// Propagates deltas back through the pooling layer.
    /// The chosen max value is saved in each forward pass and used in this backward pass to determine the neuron that will receive the delta.
    /// </summary>
    /// <description></description>
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
            MyLog.DEBUG.WriteLine("Pooling backward.");
            
            // pointer to previous layer
            MyNode node = Owner.Input.Owner;

            if (node is MyAbstractLayer)
            {
                MyAbstractLayer previousLayer = node as MyAbstractLayer;

                // reset delta
                previousLayer.Delta.Fill(0);

                // determine input to previous layer
                CUdeviceptr prevInputPtr = MyAbstractLayer.DetermineInput(previousLayer);

                m_kernel.SetupExecution(Owner.Neurons);
                m_kernel.Run(
                    (int)previousLayer.ActivationFunction,
                    Owner.Delta,
                    previousLayer.Delta,
                    prevInputPtr,
                    Owner.ActivatedNeurons,
                    Owner.Neurons
                );
            }

        }
    }


    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// Standard forward pass of the convolution operation.
    /// </summary>
    /// <description></description>
    [Description("ConvolutionForward"), MyTaskInfo(OneShot = false)]
    public class MyConvolutionForwardTask : MyAbstractForwardTask<MyConvolutionLayer>
    {
        private MyCudaKernel m_kernel, m_padKernel;

        public MyConvolutionForwardTask() { } //parameterless constructor

        public override void Init(int nGPU) //Kernel initialization
        {
            m_padKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "PadImageKernel");
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "ConvolutionForwardKernel");
        }

        public override void Execute() //Task execution
        {
            MyLog.DEBUG.WriteLine("Convolution forward.");
            



          /*  if (Owner.Input == null)
            {
                MyLog.ERROR.WriteLine("Convolution forward error: Input to " + Owner.Name + " is null.");
                return;
            }*/


            // first perform zero padding
            m_padKernel.SetupExecution(Owner.Input.Count);
            m_padKernel.Run(
                Owner.Input,
                Owner.PaddedImage,
                Owner.InputWidth,
                Owner.ZeroPadding,
                Owner.InputWidth * Owner.InputHeight,
                (Owner.InputWidth + Owner.ZeroPadding + Owner.ZeroPadding) * (Owner.InputHeight + Owner.ZeroPadding + Owner.ZeroPadding),
                Owner.Input.Count
            );





            m_kernel.SetupExecution(Owner.Output.Count);
            m_kernel.Run(
                (int)Owner.ActivationFunction,
                Owner.PaddedImage,
                Owner.Weights,
                Owner.Bias,
                Owner.Output,
                Owner.NeuronInput,
                Owner.FilterWidth, Owner.FilterHeight, Owner.InputDepth,
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

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// Computes deltas of the previous layer from deltas on this convolutional layer.
    /// </summary>
    /// <description></description>
    [Description("ConvolutionBackward"), MyTaskInfo(OneShot = false)]
    public class MyConvolutionBackwardTask : MyAbstractBackDeltaTask<MyConvolutionLayer>
    {
        private MyCudaKernel m_kernel;

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "ConvolutionBackwardKernel");
        }

        public override void Execute()
        {
            MyLog.DEBUG.WriteLine("Convolution backward.");

            // pointer to previous layer
            MyNode node = Owner.Input.Owner;

            if (node is MyAbstractLayer)
            {
                MyAbstractLayer previousLayer = node as MyAbstractLayer;

                // reset delta
                previousLayer.Delta.Fill(0);

                // determine input to previous layer
                CUdeviceptr prevInputPtr = MyAbstractLayer.DetermineInput(previousLayer);

                m_kernel.SetupExecution(previousLayer.Neurons);
                m_kernel.Run(
                    (int)previousLayer.ActivationFunction,
                    Owner.Weights,
                    Owner.Delta,
                    previousLayer.Delta,
                    prevInputPtr,
                    Owner.FilterCount,
                    Owner.InputWidth * Owner.InputHeight, // input slice size without padding
                    (Owner.InputWidth + Owner.ZeroPadding + Owner.ZeroPadding) * (Owner.InputHeight + Owner.ZeroPadding + Owner.ZeroPadding), // input slice size
                    Owner.ZeroPadding,
                    Owner.InputWidth, Owner.InputHeight,
                    Owner.FilterWidth, Owner.FilterHeight,
                    Owner.FilterWidth * Owner.FilterHeight,
                    Owner.FilterWidth * Owner.FilterHeight * Owner.InputDepth, 
                    Owner.OutputWidth, Owner.OutputHeight, Owner.OutputWidth * Owner.OutputHeight,
                    Owner.HorizontalStride, Owner.VerticalStride,
                    previousLayer.Neurons
                );
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// Randomly initialises weights and biases of the convolution layer.
    /// Uses normal distribution with standard deviation of 1 / (sqrt(input.Count))
    /// </summary>
    /// <description></description>
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

            float stdDev = 0.01f;

            // init random weights
            if (Owner.Input != null && Owner.Input.Count > 0)
                stdDev = 1.0f / (float)Math.Sqrt(Owner.Input.Count + 1);
                
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Weights.GetDevice(Owner), 0, stdDev);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Bias.GetDevice(Owner), 0, stdDev);
        }
    }


    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// Updates the weights (filters) of this convolutional layer. The exact algorithm to be used is chosen using the parent network's settings.
    /// </summary>
    /// <description></description>
    [Description("UpdateWeights"), MyTaskInfo(OneShot = false)]
    public class MyConvolutionUpdateWeights : MyAbstractUpdateWeightsTask<MyConvolutionLayer>
    {
        public override void Init(int nGPU) { }

        public override void Execute() //Task execution
        {
            // get enabled loss function
            MyTask task = Owner.ParentNetwork.GetEnabledTask("BackPropagation");
            MyAbstractBackpropTask backpropTask = null;
            if (task is MyAbstractBackpropTask)
                backpropTask = task as MyAbstractBackpropTask;
            else
                MyLog.ERROR.WriteLine("Backprop task does not derive from MyAbstractBackpropTask in " + Owner.ParentNetwork);

            if (backpropTask == null)
                MyLog.ERROR.WriteLine("Undetermined backprop task in " + Owner.ParentNetwork);
            else
            {
                backpropTask.Execute(Owner); // call the group task to do the backpropagation
            }
        }
    }
}
