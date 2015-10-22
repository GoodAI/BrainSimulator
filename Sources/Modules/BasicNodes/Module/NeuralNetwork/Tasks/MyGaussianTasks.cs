using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.RBM;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using GoodAI.Core.Nodes;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    [Description("Init"), MyTaskInfo(OneShot = true)]
    public class MyGaussianInitTask : MyTask<MyGaussianHiddenLayer>
    {
        private MyCudaKernel m_resetPriorStats;

        public MyGaussianInitTask() { }

        public override void Init(int nGPU)
        {
            m_resetPriorStats = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "GaussianResetPriorStats");
            m_resetPriorStats.SetupExecution(Owner.Input.Count);
        }

        public override void Execute()
        {
            m_resetPriorStats.Run(Owner.Input.Count, Owner.PriorGaussHiddenStatesMin, Owner.PriorGaussHiddenStatesMax);

            // fill constant sigma memory block with selected constant
            if (Owner.SigmaConstants.Count > 0)
            {
                for (int i = 0; i < Owner.SigmaConstants.Count; i++)
			    {
                    Owner.SigmaConstants.Host[i] = Owner.SigmaConstant;
			    }
                Owner.SigmaConstants.SafeCopyToDevice();
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mbr</meta>
    /// <status>Development</status>
    /// <summary>
    /// Tasks for Gaussian hidden layer.
    /// </summary>
    /// <description></description>
    [Description("FeedForward"), MyTaskInfo(OneShot = false)]
    public class MyGaussianForwardTask : MyAbstractForwardTask<MyGaussianHiddenLayer>
    {
        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tRegularization")]
        public bool ComputeRegularizationLoss { get; set; }

        float OriginalNeuralNetTrainingRate;

        public MyGaussianForwardTask() { }

        private MyCudaKernel m_forwardSamplingKernel;
        private MyCudaKernel m_resetPriorStats;
        private MyCudaKernel m_minMaxField;
        private MyCudaKernel m_samplePrior;
        private MyCudaKernel m_L1TermKernel;
        private MyCudaKernel m_L2TermKernel;
        private MyCudaKernel m_regularizationKernel;

        [MyBrowsable, Category("Generate"), YAXSerializableField(DefaultValue = false), Description("Reset Min, Max Blocks.")]
        public bool ResetPriorStats { get; set; }

        public override void Init(int nGPU)
        {
            m_forwardSamplingKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "GaussianForwardSamplingKernel");
            m_forwardSamplingKernel.SetupExecution(Owner.Neurons);

            m_resetPriorStats = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "GaussianResetPriorStats");
            m_resetPriorStats.SetupExecution(Owner.Input.Count);

            m_minMaxField = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "GaussianMinMaxField");
            m_minMaxField.SetupExecution(Owner.Input.Count);

            m_samplePrior = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "GaussianSamplePrior");
            m_samplePrior.SetupExecution(Owner.Input.Count);

            m_L1TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L1TermKernel");
            m_L1TermKernel.SetupExecution(m_L1TermKernel.MAX_THREADS);
            m_L1TermKernel.DynamicSharedMemory = m_L1TermKernel.BlockDimensions.x * sizeof(float);

            m_L2TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L2TermKernel");
            m_L2TermKernel.SetupExecution(m_L2TermKernel.MAX_THREADS);
            m_L2TermKernel.DynamicSharedMemory = m_L2TermKernel.BlockDimensions.x * sizeof(float);
            
            m_regularizationKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "GaussianRegularizationKernel");
            m_regularizationKernel.SetupExecution(m_regularizationKernel.MAX_THREADS);
            m_regularizationKernel.DynamicSharedMemory = m_regularizationKernel.BlockDimensions.x * sizeof(float);
        }
            
        public override void Execute()
        {
            // set locations for means
            CUdeviceptr means = Owner.Input.GetDevicePtr(Owner, 0);
            // set locations for sigmas (prev layer or constant
            CUdeviceptr sigmas;
            if (Owner.UseSigmaConstant)
                sigmas = Owner.SigmaConstants.GetDevicePtr(Owner);
            else
                sigmas = Owner.Input.GetDevicePtr(Owner, Owner.Input.Count / 2);

            //Owner.RandomNormal.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.Output.Count);

            // small HACK
            if (ResetPriorStats)
            {
                m_resetPriorStats.Run(Owner.Input.Count, Owner.PriorGaussHiddenStatesMin, Owner.PriorGaussHiddenStatesMax);
                ResetPriorStats = !ResetPriorStats;
            }
            // another snall HACK, store tr.rate the training rate if it is nonzero..
            if ((Owner.Parent as MyNeuralNetworkGroup).SGD.TrainingRate > 0)
            {
                OriginalNeuralNetTrainingRate = (Owner.Parent as MyNeuralNetworkGroup).SGD.TrainingRate;
            }
            // If signal on, do generation, so overwrite the Input
            if (Owner.Generate.IsIncomingRised())
            {
                // Generate random uniform <0,1> sample
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.RandomUniform.GetDevice(Owner));

                // Sample from learned prior distribution using adapted means and sigmas boundary intervals
                // Overwrite Input with sample, so subsequent feedforward will use sample
                m_samplePrior.Run(
                    Owner.Input,
                    Owner.Input.Count,
                    Owner.PriorGaussHiddenStatesMin,
                    Owner.PriorGaussHiddenStatesMax,
                    Owner.RandomUniform
                );
            }
            else
            {
                (Owner.Parent as MyNeuralNetworkGroup).SGD.TrainingRate = OriginalNeuralNetTrainingRate;
            }

            // Generate random normal N(0,1) sample
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.RandomNormal.GetDevice(Owner), 0, 1);

            // Generate noisy input by: NoisyInput = mean + RandomNormal * sigma
            m_forwardSamplingKernel.Run(
                (int)Owner.ActivationFunction,
                means,
                sigmas,
                Owner.NoisyInput,
                Owner.Output,
                Owner.RandomNormal,
                Owner.Input.Count,
                Owner.Output.Count
            );

            // calculate prior for generation
            m_minMaxField.Run(
                Owner.Input,
                Owner.Input.Count,
                Owner.PriorGaussHiddenStatesMin,
                Owner.PriorGaussHiddenStatesMax
            );

            if (Owner.ParentNetwork.L1 > 0) // don't take performance hit if L1 is not used
            {
                m_L1TermKernel.Run(
                    Owner.Weights,
                    Owner.L1Term,
                    Owner.Weights.Count
                );
            }

            if (Owner.ParentNetwork.L2 > 0) // don't take performance hit if L2 is not used
            {
                m_L2TermKernel.Run(
                    Owner.Weights,
                    Owner.L2Term,
                    Owner.Weights.Count
                );
            }

            if (ComputeRegularizationLoss)
            {
                m_regularizationKernel.Run(
                    means,
                    sigmas,
                    Owner.Input.Count,
                    Owner.Regularization
                );
            }
        }
    }

    /// <author>GoodAI</author>
    /// <meta>mbr</meta>
    /// <status>Development</status>
    /// <summary>
    /// Backpropagate the deltas first from Gaussians to parameters (mu, sigma) and then from parameters to input
    /// </summary>
    /// <description></description>
    [Description("DeltaBack"), MyTaskInfo(OneShot = false)]
    public class MyGaussianBackDeltaTask : MyAbstractBackDeltaTask<MyGaussianHiddenLayer>
    {
        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("Regularization")]
        public bool Regularize { get; set; }

        [YAXSerializableField(DefaultValue = 1.0f)]
        [MyBrowsable, Category("Regularization")]
        public float RegularizationCoefficient { get; set; }

        public MyGaussianBackDeltaTask() { }

        private MyCudaKernel m_samplingDeltaKernel;
        private MyCudaKernel m_regularizationDeltaKernel;

        public override void Init(int nGPU)
        {
            m_samplingDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "GaussianSamplingDeltaKernel");
            m_samplingDeltaKernel.SetupExecution(Owner.Neurons);

            m_regularizationDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "GaussianRegularizationDeltaKernel");
        }

        public override void Execute()
        {
            MyNode node = Owner.Input.Owner;

            if (node is MyAbstractLayer)
            {
                MyAbstractLayer previousLayer = node as MyAbstractLayer;

                // Reset delta
                previousLayer.Delta.Fill(0);

                // Disable backprop when in generative mode
                if (!Owner.Generate.IsIncomingRised())
                {
                    // Set locations for mean deltas
                    CUdeviceptr meanDeltas = previousLayer.Delta.GetDevicePtr(Owner, 0);
                    // Set locations for sigma deltas
                    CUdeviceptr sigmaDeltas = previousLayer.Delta.GetDevicePtr(Owner, previousLayer.Delta.Count / 2);
                    // Determine input to previous layer
                    CUdeviceptr prevInputPtr = MyAbstractLayer.DetermineInput(previousLayer);
                    // set locations for sigmas (prev layer or constant
                    CUdeviceptr sigmas;
                    if (Owner.UseSigmaConstant)
                        sigmas = Owner.SigmaConstants.GetDevicePtr(Owner);
                    else
                        sigmas = Owner.Input.GetDevicePtr(Owner, Owner.Input.Count / 2);

                    m_samplingDeltaKernel.Run(
                        Convert.ToInt32(Owner.UseSigmaConstant),
                        (int)previousLayer.ActivationFunction,
                        prevInputPtr,
                        sigmas,
                        meanDeltas,
                        sigmaDeltas,
                        Owner.Delta,
                        Owner.RandomNormal,
                        Owner.Neurons
                    );

                    // Regularization needs weights to compute gradients
                    if (Regularize && previousLayer != null && previousLayer is MyAbstractWeightLayer)
                    {
                        MyAbstractWeightLayer previousWeightLayer = previousLayer as MyAbstractWeightLayer;

                        // Try to regularize loss: mean^2 + sigma^2 - log(sigma^2)
                        // In other words regularize means to 0 and sigmas to 1
                        int weightCount = previousWeightLayer.Weights.Count;
                        m_regularizationDeltaKernel.SetConstantVariable<float>("RegularizationCoefficient", RegularizationCoefficient);
                        m_regularizationDeltaKernel.SetupExecution(weightCount);
                        m_regularizationDeltaKernel.Run(
                            Convert.ToInt32(Owner.UseSigmaConstant),
                            (int)previousLayer.ActivationFunction,
                            prevInputPtr,
                            previousLayer.Input,
                            previousWeightLayer.Weights,
                            previousLayer.Output.Count,
                            meanDeltas,
                            sigmaDeltas
                        );
                    }
                }
            }
        }
    }
}
