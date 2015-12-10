using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using GoodAI.Modules.Matrix;
using GoodAI.Modules.NeuralNetwork.Layers;
using System;
using System.ComponentModel;
using System.Collections;
using System.Linq;
using YAXLib;
using System.Reflection;
using System.Collections.Generic;

namespace GoodAI.Modules.NeuralNetwork.Group
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Initialises Neural Network Group and sets internally used properties.</summary>
    /// <description></description>
    [Description("InitGroup"), MyTaskInfo(OneShot = true)]
    public class MyInitNNGroupTask : MyTask<MyNeuralNetworkGroup>
    {
        // parameterless constructor
        public MyInitNNGroupTask() { }

        // Kernel initialization
        public override void Init(int nGPU) { }

        // Task execution
        public override void Execute()
        {
            // timeStep is -1, because it is incremented at beginning of new timestep
            Owner.TimeStep = -1;

            // disable GradientCheck by default - TODO: fix this somehow
            Owner.GradientCheck.Enabled = false;

            // sort children in topological order
            Owner.SortedChildren = Owner.Children.OrderBy(o => o.TopologicalOrder).ToList();

            // set next and previous layer
            MyAbstractLayer layer;
            MyAbstractLayer lastTopologicalLayer = null;
            foreach (MyNode child in Owner.SortedChildren)
            {
                if (child is MyAbstractLayer)
                {
                    layer = child as MyAbstractLayer;

                    if (lastTopologicalLayer != null)
                    {
                        lastTopologicalLayer.NextTopologicalLayer = layer;
                    }

                    layer.NextTopologicalLayer = null;
                    layer.PreviousTopologicalLayer = lastTopologicalLayer;
                    lastTopologicalLayer = layer;

                    // collect all next and all previous conneted layers
                    layer.PreviousConnectedLayers = new List<MyAbstractLayer>();
                    layer.NextConnectedLayers = new List<MyAbstractLayer>();
                    foreach (MyConnection inputConnection in child.InputConnections)
                    {
                        if (inputConnection != null && inputConnection.From is MyAbstractLayer)
                        {
                            MyAbstractLayer lastConnectedLayer = inputConnection.From as MyAbstractLayer;
                            layer.PreviousConnectedLayers.Add(lastConnectedLayer);
                            lastConnectedLayer.NextConnectedLayers.Add(layer);
                        }
                    }
                }
            }

            // set first and last layer
            layer = lastTopologicalLayer;
            Owner.FirstTopologicalLayer = layer;
            Owner.LastTopologicalLayer = layer;
            while (layer != null)
            {
                Owner.FirstTopologicalLayer = layer;
                layer = layer.PreviousTopologicalLayer;
            }

            // count total number of weights
            Owner.TotalWeights = 0;
            layer = Owner.FirstTopologicalLayer;
            while (layer != null)
            {
                if (layer is MyAbstractWeightLayer)
                    Owner.TotalWeights += (layer as MyAbstractWeightLayer).Weights.Count;
                layer = layer.NextTopologicalLayer;
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>
    /// Used only if sequence length (neural group parameter) > 1.
    /// Increments time step when iterating forth through time.
    /// Used automatically, keep it checked.
    /// </summary>
    /// <description></description>
    [Description("IncrementTimeStep"), MyTaskInfo(OneShot = false)]
    public class MyIncrementTimeStepTask : MyTask<MyNeuralNetworkGroup>
    {
        public MyIncrementTimeStepTask() { }

        public override void Init(int nGPU) { }

        public override void Execute()
        {
            Owner.TimeStep++;
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>
    /// Used only if sequence length (neural group parameter) > 1.
    /// Decrements time step when iterating back through time.
    /// Used automatically, keep it checked.
    /// </summary>
    /// <description></description>
    [Description("DecrementTimeStep"), MyTaskInfo(OneShot = false)]
    public class MyDecrementTimeStepTask : MyTask<MyNeuralNetworkGroup>
    {
        public MyDecrementTimeStepTask() { }

        public override void Init(int nGPU) { }

        public override void Execute()
        {
            Owner.TimeStep--;
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mb</meta>
    /// <status>Working</status>
    /// <summary>
    /// Used only if sequence length (neural group parameter) > 1. When time step reaches the end of sequence,
    /// then temporal memory blocks run a mode (method), e.g. sum memory snapshots through time. Default mode is none.
    /// New modes can be defined in temporal memory blocks source. Used automatically, let it checked.
    /// </summary>
    /// <description></description>
    [Description("RunTemporalBlocksMode"), MyTaskInfo(OneShot = false)]
    public class MyRunTemporalBlocksModeTask : MyTask<MyNeuralNetworkGroup>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            foreach (MyNode child in Owner.SortedChildren)
            {
                if (child is MyAbstractLayer)
                {
                    foreach (PropertyInfo memBlockInfo in MyNodeInfo.Get(child.GetType()).OwnedMemoryBlocks)
                    {
                        Object memBlock = memBlockInfo.GetValue(child);
                        if (memBlock is MyTemporalMemoryBlock<float>)
                        {
                            (memBlock as MyTemporalMemoryBlock<float>).RunMode();
                        }
                    }
                }
            }
        }
    }

    public abstract class MyAbstractBackpropTask : MyTask<MyNeuralNetworkGroup>
    {
        public virtual void Execute(MyAbstractLayer layer)
        {
            MyLog.ERROR.WriteLine("No method provided to backpropagate MyAbstractLayer " + layer + " in " + Owner);
        }

        public virtual void Execute(MyAbstractWeightLayer layer)
        {
            MyLog.ERROR.WriteLine("No method provided to backpropagate MyAbstractWeightLayer " + layer + " in " + Owner);
        }

        public void ComputeWeightGradientSum(MyAbstractWeightLayer layer)
        {
            if (Owner.BatchSize == 1) // cuBLAS tends to be slower when BatchSize is 1, gradient is computed in update weights kernels
                return;

            // WeightGradient = Delta x Transpose(Input)
            MyCublasFactory.Instance.Gemm(Operation.NonTranspose, Operation.Transpose,
                layer.Neurons, layer.Input.Count / Owner.BatchSize, Owner.BatchSize, 1.0f,
                layer.Delta.GetDevice(layer), layer.Neurons,
                layer.Input.GetDevice(layer), layer.Input.Count / Owner.BatchSize,
                0.0f, layer.WeightGradient.GetDevice(layer), layer.Neurons
                );
            
            // BiasGradient = Delta x Transpose(BiasInput). BiasInput is vector of ones
            MyCublasFactory.Instance.Gemm(Operation.NonTranspose, Operation.Transpose,
                layer.Neurons, 1, Owner.BatchSize, 1.0f,
                layer.Delta.GetDevice(layer), layer.Neurons,
                layer.BiasInput.GetDevice(layer), 1,
                0.0f, layer.BiasGradient.GetDevice(layer), layer.Neurons
                );
        }

        [YAXSerializableField(DefaultValue = false), MyBrowsable, Description("Disables and overrides layers' UpdateWeights checking if and only if set to TRUE. No effect when set to FALSE.")]
        public bool DisableLearning { get; set; }
    }

    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Stochastic gradient descent is an online training algorithm, that updates each parameter in the direction of the gradient for the current training example.<br></br>
    /// The gradient used is the partial derivative of each parameter (weight or bias) with respect to the loss function.</summary>
    /// <description></description>
    [Description("StochasticGradientDescent"), MyTaskInfo(OneShot = false)]
    public class MySGDTask : MyAbstractBackpropTask
    {
        // properties
        [YAXSerializableField(DefaultValue = 0.25f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float TrainingRate { get; set; }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float Momentum { get; set; }

        //parameterless constructor
        public MySGDTask() { }

        // kernel
        private MyCudaKernel m_SGDupdateKernel, m_convSGDupdateKernel, m_partialSGDupdateKernel;
        public override void Init(int nGPU)
        {
            m_SGDupdateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\UpdateWeightsKernels", "FullyConnectedSGDUpdateKernel");
            m_convSGDupdateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "ConvolutionSGDUpdateWeightsKernel");
            m_partialSGDupdateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\UpdateWeightsKernels", "PartialSGDUpdateKernel");
        }

        //Task execution - should be called with a parameter
        public override void Execute()
        {
            //MyLog.ERROR.WriteLine("Please Execute MySGDTask with a layer parameter in " + Owner);
        }

        public override void Execute(MyAbstractWeightLayer layer)
        {
            if (layer.Connection == ConnectionType.FULLY_CONNECTED)
            {
                ComputeWeightGradientSum(layer);
                
                m_SGDupdateKernel.SetupExecution(layer.Weights.Count);
                m_SGDupdateKernel.Run(
                    layer.Input,
                    layer.Delta,
                    layer.WeightGradient,
                    layer.BiasGradient,
                    layer.Weights,
                    layer.PreviousWeightDelta,
                    layer.Bias,
                    layer.PreviousBiasDelta,
                    Owner.SGD.TrainingRate,
                    Owner.SGD.Momentum,
                    Owner.L1,
                    Owner.L2,
                    layer.DropoutMask,
                    layer.Neurons,
                    Owner.BatchSize,
                    layer.Weights.Count
                    );

            }
            else if (layer.Connection == ConnectionType.GAUSSIAN)
            {
                // Gaussian hidden layer just propagates delta, no weight updates
            }
            else if (layer.Connection == ConnectionType.PARTIAL_UPDATE && layer is IPartialUpdateLayer)
            {
                // Update some but not all of the weights
                IPartialUpdateLayer partialUpdateLayer = layer as IPartialUpdateLayer;

                m_partialSGDupdateKernel.SetupExecution(layer.Weights.Count);
                m_partialSGDupdateKernel.Run(
                    layer.Input,
                    layer.Delta,
                    layer.Weights,
                    layer.PreviousWeightDelta,
                    layer.Bias,
                    layer.PreviousBiasDelta,
                    Owner.SGD.TrainingRate,
                    Owner.SGD.Momentum,
                    Owner.L1,
                    Owner.L2,
                    layer.DropoutMask,
                    layer.Neurons,
                    layer.Weights.Count,
                    partialUpdateLayer.SuppressUpdatesAt(),
                    partialUpdateLayer.SuppressUpdatesCount()
                );
            }
            else if (layer.Connection == ConnectionType.CONVOLUTION && layer is MyConvolutionLayer)
            {
                MyConvolutionLayer convLayer = (MyConvolutionLayer) layer;
                m_convSGDupdateKernel.SetupExecution(convLayer.Weights.Count);
                m_convSGDupdateKernel.Run(
                    Owner.SGD.TrainingRate, Owner.SGD.Momentum,
                    convLayer.Weights,
                    convLayer.Bias, convLayer.PreviousBiasDelta,
                    convLayer.Delta, convLayer.PreviousWeightDelta,
                    convLayer.PaddedImage,
                    convLayer.InputWidth + convLayer.ZeroPadding + convLayer.ZeroPadding,
                    (convLayer.InputWidth + convLayer.ZeroPadding + convLayer.ZeroPadding)*
                    (convLayer.InputHeight + convLayer.ZeroPadding + convLayer.ZeroPadding),
                    convLayer.FilterWidth,
                    convLayer.FilterWidth*convLayer.FilterHeight,
                    convLayer.FilterWidth*convLayer.FilterHeight*convLayer.InputDepth,
                    convLayer.OutputWidth, convLayer.OutputHeight, convLayer.OutputWidth*convLayer.OutputHeight,
                    convLayer.HorizontalStride, convLayer.VerticalStride,
                    convLayer.L1Term, convLayer.L2Term,
                    Owner.BatchSize,
                    convLayer.Weights.Count
                    // should be equal to FilterWidth * FilterHeight * FilterCount * InputDepth
                    );
            }
            else
            {
                MyLog.ERROR.WriteLine("No method provided to SGD propagate a " + layer.Connection +
                                        " connected MyAbstractWeightLayer in " + Owner);
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>RMSProp is the online adaptation of the Resilient Backpropagation algorithm based on the mean squares of the parameters.<br></br>
    /// It solves the problem of saturated neurons and vanishing gradients, which can occur with other backpropagation methods.<br></br>
    /// The mean squares are moving averages based on the smoothing factor, so as to emulate batch learning.</summary>
    /// <description></description>
    [Description("RMSProp"), MyTaskInfo(OneShot = false)]
    public class MyRMSTask : MyAbstractBackpropTask
    {
        // properties
        [YAXSerializableField(DefaultValue = 0.0025f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float TrainingRate { get; set; }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float Momentum { get; set; }

        [YAXSerializableField(DefaultValue = 0.9f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float SmoothingFactor { get; set; }

        //parameterless constructor
        public MyRMSTask() { }

        //Kernel initialization
        private MyCudaKernel m_RMSPropUpdateKernel, m_convRMSPropUpdateKernel;
        public override void Init(int nGPU)
        {
            m_RMSPropUpdateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\UpdateWeightsKernels", "FullyConnectedRMSPropUpdateKernel");
            m_convRMSPropUpdateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "ConvolutionRMSPropUpdateWeightsKernel");
        }

        //Task execution
        public override void Execute()
        {
            //MyLog.ERROR.WriteLine("Please Execute MyRMSTask with a layer parameter in " + Owner);
        }

        public override void Execute(MyAbstractWeightLayer layer)
        {
            if (layer.Connection == ConnectionType.FULLY_CONNECTED)
            {
                ComputeWeightGradientSum(layer);

                m_RMSPropUpdateKernel.SetupExecution(layer.Weights.Count);
                m_RMSPropUpdateKernel.Run(
                    layer.Input,
                    layer.Delta,
                    layer.WeightGradient,
                    layer.BiasGradient,
                    layer.Weights,
                    layer.PreviousWeightDelta,
                    layer.Bias,
                    layer.PreviousBiasDelta,
                    Owner.RMS.TrainingRate,
                    Owner.RMS.Momentum,
                    Owner.L1,
                    Owner.L2,
                    layer.DropoutMask,
                    layer.Neurons,
                    Owner.BatchSize,
                    layer.Weights.Count,
                    layer.MeanSquareWeight,
                    layer.MeanSquareBias,
                    Owner.RMS.SmoothingFactor
                    );
            }
            else if (layer.Connection == ConnectionType.GAUSSIAN)
            {
                // Gaussian hidden layer just propagates delta, no weight updates
            }
            else if (layer.Connection == ConnectionType.CONVOLUTION && layer is MyConvolutionLayer)
            {
                MyConvolutionLayer convLayer = (MyConvolutionLayer)layer;
                m_convRMSPropUpdateKernel.SetupExecution(convLayer.Weights.Count);
                m_convRMSPropUpdateKernel.Run(
                    Owner.RMS.TrainingRate, Owner.RMS.Momentum,
                    convLayer.Weights,
                    convLayer.Bias, convLayer.PreviousBiasDelta,
                    convLayer.Delta, convLayer.PreviousWeightDelta,
                    convLayer.PaddedImage,
                    convLayer.InputWidth + convLayer.ZeroPadding + convLayer.ZeroPadding,
                    (convLayer.InputWidth + convLayer.ZeroPadding + convLayer.ZeroPadding) *
                    (convLayer.InputHeight + convLayer.ZeroPadding + convLayer.ZeroPadding),
                    convLayer.FilterWidth,
                    convLayer.FilterWidth * convLayer.FilterHeight,
                    convLayer.FilterWidth * convLayer.FilterHeight * convLayer.InputDepth,
                    convLayer.OutputWidth, convLayer.OutputHeight, convLayer.OutputWidth * convLayer.OutputHeight,
                    convLayer.HorizontalStride, convLayer.VerticalStride,
                    convLayer.L1Term, convLayer.L2Term,
                    convLayer.MeanSquareWeight, convLayer.MeanSquareBias, Owner.RMS.SmoothingFactor,
                    Owner.BatchSize,
                    convLayer.Weights.Count
                    // should be equal to FilterWidth * FilterHeight * FilterCount * InputDepth
                    );
            }
            else
            {
                MyLog.ERROR.WriteLine("No method provided to RMS propagate a " + layer.Connection +
                                      " connected MyAbstractWeightLayer in " + Owner);
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    ///     Adadelta is an adaptive learning method that changes each weight (parameter) separately and automatically over time.<br></br>
    ///     No manual settings are needed and it is recommended to use the default values which should behave well in all cases.
    /// </summary>
    [Description("Adadelta"), MyTaskInfo(OneShot = false)]
    public class MyAdadeltaTask : MyAbstractBackpropTask
    {
        // properties
        [YAXSerializableField(DefaultValue = 0.000001f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float Epsilon { get; set; }

        [YAXSerializableField(DefaultValue = 0.95f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float Ro { get; set; }

        //Kernel initialization
        private MyCudaKernel m_adadeltaUpdateKernel, m_convAdadeltaUpdateKernel;
        public override void Init(int nGPU)
        {
            m_adadeltaUpdateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\UpdateWeightsKernels", "FullyConnectedAdadeltaUpdateKernel");
            m_convAdadeltaUpdateKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Convolution\ConvolutionKernel", "ConvolutionAdadeltaUpdateWeightsKernel");
        }

        //Task execution
        public override void Execute()
        {
            //MyLog.ERROR.WriteLine("Please Execute Adadelta with a layer parameter in " + Owner);
        }

        public override void Execute(MyAbstractWeightLayer layer)
        {
            if (layer.Connection == ConnectionType.FULLY_CONNECTED)
            {
                ComputeWeightGradientSum(layer);

                m_adadeltaUpdateKernel.SetupExecution(layer.Weights.Count);
                m_adadeltaUpdateKernel.Run(
                    layer.Input,
                    layer.Delta,
                    layer.WeightGradient,
                    layer.BiasGradient,
                    layer.Weights,
                    layer.Bias,
                    Owner.L1,
                    Owner.L2,
                    layer.DropoutMask,
                    layer.Neurons,
                    Owner.BatchSize,
                    layer.Weights.Count,
                    layer.MeanSquareWeight, layer.PreviousWeightDelta, layer.MeanSquareBias, layer.PreviousBiasDelta,
                    Owner.Adadelta.Ro, Owner.Adadelta.Epsilon
                    );
            }
            else if (layer.Connection == ConnectionType.GAUSSIAN)
            {
                // Gaussian hidden layer just propagates delta, no weight updates
            }
            else if (layer.Connection == ConnectionType.CONVOLUTION && layer is MyConvolutionLayer)
            {
                MyConvolutionLayer convLayer = (MyConvolutionLayer)layer;
                m_convAdadeltaUpdateKernel.SetupExecution(convLayer.Weights.Count);
                m_convAdadeltaUpdateKernel.Run(
                    convLayer.Weights,
                    convLayer.Bias,
                    convLayer.Delta,
                    convLayer.PaddedImage,
                    convLayer.InputWidth + convLayer.ZeroPadding + convLayer.ZeroPadding,
                    (convLayer.InputWidth + convLayer.ZeroPadding + convLayer.ZeroPadding) *
                    (convLayer.InputHeight + convLayer.ZeroPadding + convLayer.ZeroPadding),
                    convLayer.FilterWidth,
                    convLayer.FilterWidth * convLayer.FilterHeight,
                    convLayer.FilterWidth * convLayer.FilterHeight * convLayer.InputDepth,
                    convLayer.OutputWidth, convLayer.OutputHeight, convLayer.OutputWidth * convLayer.OutputHeight,
                    convLayer.HorizontalStride, convLayer.VerticalStride,
                    convLayer.L1Term, convLayer.L2Term,
                    convLayer.MeanSquareWeight, convLayer.PreviousWeightDelta, convLayer.MeanSquareBias,
                    convLayer.PreviousBiasDelta,
                    Owner.Adadelta.Ro, Owner.Adadelta.Epsilon,
                    Owner.BatchSize,
                    convLayer.Weights.Count
                    // should be equal to FilterWidth * FilterHeight * FilterCount * InputDepth
                    );
            }
            else
            {
                MyLog.ERROR.WriteLine("No method provided to Adadelta propagate a " + layer.Connection +
                                      " connected MyAbstractWeightLayer in " + Owner);
            }
        }
    }


    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Gradient checking mainly for developers to make sure the calculated gradients are correct.
    /// <br></br>
    /// This should generally be disabled during training, since it will negatively affect performance</summary>
    /// <description></description>
    [Description("GradientCheck"), MyTaskInfo(OneShot = false)]
    public class MyGradientCheckTask : MyTask<MyNeuralNetworkGroup>
    {
        //Properties
        [YAXSerializableField(DefaultValue = 0.1f)]
        [MyBrowsable, Category("\tParams")]
        public float RelativeStepSize { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tParams")]
        public int SamplesPerTimestep { get; set; }

        [YAXSerializableField(DefaultValue = 0.001f)]
        [MyBrowsable, Category("\tParams")]
        public float ThresholdRelative { get; set; }

        [YAXSerializableField(DefaultValue = 0.0001f)]
        [MyBrowsable, Category("\tParams")]
        public float ThresholdAbsolute { get; set; }

        private Random Rand = new Random();

        public MyGradientCheckTask() { } //parameterless constructor

        public override void Init(int nGPU) { } //Kernel initialization

        public override void Execute()
        {
            float maxRelDiff = 0.0f;
            float maxAbsDiff = 0.0f;
            int maxDiffLayer = 0;
            int maxDiffWeight = 0;
            float maxDiffWeightValue = 0.0f;
            float maxDiffStepSize = 0.0f;
            float maxDiffAnalyticalGrad = 0.0f;
            float maxDiffNumericalGrad = 0.0f;

            float sampleProbability = 1.0f / Owner.TotalWeights;
            for (int s = 0; s < SamplesPerTimestep; s++)
            {
                // dice roll
                float diceRoll = (float)Rand.NextDouble();

                // convert diceroll to parameter to sample
                int w = (int)Math.Floor(diceRoll / sampleProbability);
                if (w >= Owner.TotalWeights)
                {
                    if (w > Owner.TotalWeights)
                        MyLog.ERROR.Write("w > Owner.TotalWeights"); // just for testing, this should never hit
                    w = Owner.TotalWeights - 1; // this is just to make if safe, but it should never hit
                }

                // loop through the layers
                MyAbstractLayer layer = Owner.FirstTopologicalLayer;
                while (layer != null)
                {
                    // check for weights
                    if (layer is MyAbstractWeightLayer)
                    {
                        MyAbstractWeightLayer weightLayer = (layer as MyAbstractWeightLayer);
                        if (weightLayer.Weights.Count <= w)
                            w -= weightLayer.Weights.Count;
                        else
                        {
                            weightLayer.Weights.SafeCopyToHost(w, 1); // copy this weight to host
                            float originalWeight = weightLayer.Weights.Host[w]; // save weight
                            float stepSize = Math.Abs(originalWeight) * RelativeStepSize; // set stepSize

                            // get errorPlus
                            weightLayer.Weights.Host[w] = originalWeight + stepSize; // increase weight
                            weightLayer.Weights.SafeCopyToDevice(w, 1); // back to device
                            Owner.FeedForward(); // forward the network
                            float errorPlus = Owner.GetError();

                            // get errorMinus
                            weightLayer.Weights.Host[w] = originalWeight - stepSize; // decrease weight
                            weightLayer.Weights.SafeCopyToDevice(w, 1); // back to device
                            Owner.FeedForward(); // forward the network
                            float errorMinus = Owner.GetError();

                            // reset to original
                            weightLayer.Weights.Host[w] = originalWeight; // back to where we started
                            weightLayer.Weights.SafeCopyToDevice(w, 1); // back to device
                            Owner.FeedForward(); // forward the network
                            Owner.GetError(); // this sets the original error

                            // numerical gradient
                            float numericalGradient = (errorPlus - errorMinus) / (2 * stepSize);

                            if (numericalGradient == 0)
                            {
                                MyLog.DEBUG.WriteLine("t: " + SimulationStep + " id: " + weightLayer.Id + " w" + w + ": " + weightLayer.Weights.Host[w] + " step: " + stepSize + " numerical gradient is 0.");
                                break; // continue to next sample
                            }

                            // analytical gradient
                            int n = w % weightLayer.Neurons;
                            int i = (w - n) / weightLayer.Neurons;
                            weightLayer.Delta.SafeCopyToHost(n, 1); // copy delta to host
                            weightLayer.Input.SafeCopyToHost(i, 1); // copy input to host
                            weightLayer.DropoutMask.SafeCopyToHost(n, 1); // copy dropoutmask to host
                            //weightLayer.Weights.SafeCopyToHost(w, 1); // already present at host due to resetting to original
                            if (weightLayer.DropoutMask.Host[n] > 0)
                                break;
                            float analyticalGradient = weightLayer.Delta.Host[n] * weightLayer.Input.Host[i] + Owner.L1 * (weightLayer.Weights.Host[w] < 0.0f ? -1.0f : 1.0f) + Owner.L2 * weightLayer.Weights.Host[w];
                            float relativeDiff = 0.0f;
                            float absoluteDiff = 0.0f;
                            if (analyticalGradient == 0)
                            {
                                MyLog.DEBUG.WriteLine("t: " + SimulationStep + " id: " + weightLayer.Id + " w" + w + ": " + weightLayer.Weights.Host[w] + " step: " + stepSize + " analytical gradient is 0.");
                                break; // continue to next sample
                            }
                            absoluteDiff = Math.Abs(numericalGradient - analyticalGradient);
                            relativeDiff = absoluteDiff / (Math.Abs(numericalGradient) + Math.Abs(analyticalGradient));
                            if (relativeDiff > maxRelDiff && absoluteDiff > ThresholdAbsolute)
                            {
                                maxAbsDiff = absoluteDiff;
                                maxRelDiff = relativeDiff;
                                maxDiffLayer = weightLayer.Id;
                                maxDiffWeight = w;
                                maxDiffWeightValue = weightLayer.Weights.Host[w];
                                maxDiffStepSize = stepSize;
                                maxDiffAnalyticalGrad = analyticalGradient;
                                maxDiffNumericalGrad = numericalGradient;
                            }
                            MyLog.DEBUG.WriteLine("t: " + SimulationStep + " id: " + weightLayer.Id + " w" + w + ": " + weightLayer.Weights.Host[w] + " step: " + stepSize + " AG: " + analyticalGradient + " NG: " + numericalGradient + " diff: " + relativeDiff);
                            break; // continue to next sample
                        }
                    }
                    layer = layer.NextTopologicalLayer;

                    // catch unmatched dice-rolls
                    if (layer == null)
                        MyLog.ERROR.Write("GradientCheck task: Weight w " + w + " not found within " + Owner.TotalWeights + " total weights"); // just for testing, this should never hit
                }
            }
            // handle the relativeDiff we just found
            if (maxRelDiff > ThresholdRelative && maxRelDiff > ThresholdAbsolute)
            {
                MyLog.INFO.WriteLine("Gradient threshold exceeded on SimulationStep: " + SimulationStep);
                MyLog.INFO.WriteLine("Max analytical vs numerical relative gradient difference found in layer id " + maxDiffLayer + " for weight " + maxDiffWeight + ": " + maxDiffWeightValue + " with Step size: " + maxDiffStepSize);
                MyLog.INFO.WriteLine("Analytical gradient: " + maxDiffAnalyticalGrad + " Numerical gradient: " + maxDiffNumericalGrad + " Relative difference: " + maxRelDiff);
                MyLog.INFO.WriteLine();
            }
        }
        //        // copy delta to host
        //        selectedLayer.Delta.SafeCopyToHost();

        //        // loop the neurons
        //        for (int n = 0; n < selectedLayer.Neurons; n++)
        //        {
        //            // nudge each neurons weightedInput with MiniStepSize forward
        //            selectedLayer.WeightedInput.SafeCopyToHost(n, 1); // copy to host
        //            selectedLayer.WeightedInput.Host[n] += MiniStepSize; // increment with MiniStep
        //            selectedLayer.WeightedInput.SafeCopyToDevice(n, 1); // back to device

        //            // find lossPlus
        //            FeedForward(selectedLayer);
        //            Owner.OutputLayer.CalcDelta.Execute(); // save loss contribution
        //            Owner.Loss.Host[0] = 0;
        //            Owner.OutputLayer.LossContribution.SafeCopyToHost();
        //            for (int i = 0; i < Owner.OutputLayer.Neurons; i++)
        //                Owner.Loss.Host[0] += Owner.OutputLayer.LossContribution.Host[i];
        //            Owner.Loss.SafeCopyToDevice();
        //            float lossPlus = Owner.Loss.Host[0];

        //            // nudge each neurons weightedInput with MiniStepSize backward
        //            selectedLayer.WeightedInput.Host[n] -= 2 * MiniStepSize; // decrement 2 MiniSteps
        //            selectedLayer.WeightedInput.SafeCopyToDevice(n, 1); // copy to device

        //            // find lossMinus
        //            FeedForward(selectedLayer);
        //            Owner.OutputLayer.CalcDelta.Execute(); // save loss contribution
        //            Owner.Loss.Host[0] = 0;
        //            Owner.OutputLayer.LossContribution.SafeCopyToHost();
        //            for (int i = 0; i < Owner.OutputLayer.Neurons; i++)
        //                Owner.Loss.Host[0] += Owner.OutputLayer.LossContribution.Host[i];
        //            Owner.Loss.SafeCopyToDevice();
        //            Owner.Loss.SafeCopyToHost();
        //            float lossMinus = Owner.Loss.Host[0];

        //            // calculate numerical gradient
        //            selectedLayer.NumericalDelta.Host[n] = (lossPlus - lossMinus) / (2 * MiniStepSize);
        //            selectedLayer.NumericalDelta.SafeCopyToDevice(n, 1); // copy to device

        //            if (selectedLayer.NumericalDelta.Host[n] == 0 || selectedLayer.Delta.Host[n] == 0)
        //                Owner.ZeroValuedDeltas.Host[0]++; // count neurons with zero delta on host
        //            else
        //            {
        //                float deltaDiff = Math.Abs(selectedLayer.Delta.Host[n] - selectedLayer.NumericalDelta.Host[n]) / (Math.Abs(selectedLayer.NumericalDelta.Host[n]) + Math.Abs(selectedLayer.Delta.Host[n]));// set max delta diff
        //                Owner.AvgRelativeDeltaDiff.Host[0] += deltaDiff;
        //                if (deltaDiff > Owner.MaxRelativeDeltaDiff.Host[0])
        //                    Owner.MaxRelativeDeltaDiff.Host[0] = deltaDiff;
        //                neurons++;
        //            }

        //            // reset input
        //            selectedLayer.WeightedInput.Host[n] += MiniStepSize; // increment with MiniStep
        //            selectedLayer.WeightedInput.SafeCopyToDevice(n, 1); // copy to device
        //        }

        //        // select next layer
        //        selectedLayer = selectedLayer.NextLayer as MyLayerfloat; // this could potentially break, if there is an abstract layer present (which there should not be)
        //    }

        //    // copy saturated neurons to device
        //    Owner.ZeroValuedDeltas.SafeCopyToDevice();

        //    // calc and copy avg delta diff to device
        //    Owner.AvgRelativeDeltaDiff.Host[0] /= neurons;
        //    Owner.AvgRelativeDeltaDiff.SafeCopyToDevice();

        //    Owner.MaxRelativeDeltaDiff.SafeCopyToDevice();

        //    // cast to float value
        //    Owner.AvgRelativeDeltaDiffFloat.Host[0] = (float)Owner.AvgRelativeDeltaDiff.Host[0];
        //    Owner.AvgRelativeDeltaDiffFloat.SafeCopyToDevice();

        //    Owner.MaxRelativeDeltaDiffFloat.Host[0] = (float)Owner.MaxRelativeDeltaDiff.Host[0];
        //    Owner.MaxRelativeDeltaDiffFloat.SafeCopyToDevice();
        //}
    }

}