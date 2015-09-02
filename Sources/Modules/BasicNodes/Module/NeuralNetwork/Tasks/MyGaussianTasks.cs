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

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    [Description("Init"), MyTaskInfo(OneShot = true)]
    public class MyGaussianInitTask : MyTask<MyGaussianHiddenLayer>
    {
        public MyGaussianInitTask() { }

        public override void Init(int nGPU) { }

        public override void Execute()
        {
            Owner.ResetPriorStats();

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
        Random rnd = new Random();
        float OriginalNeuralNetTrainingRate;

        public MyGaussianForwardTask() { }

        private MyCudaKernel m_forwardSamplingKernel;
        private MyCudaKernel m_L1TermKernel;
        private MyCudaKernel m_L2TermKernel;
        private MyCudaKernel m_gaussianRegularizationKernel;

        [MyBrowsable, Category("Generate"), YAXSerializableField(DefaultValue = false), Description("Reset Min, Max Blocks.")]
        public bool ResetPriorStats { get; set; }

        public override void Init(int nGPU)
        {
            m_forwardSamplingKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "GaussianForwardSamplingKernel");
            m_forwardSamplingKernel.SetupExecution(Owner.Neurons);

            m_L1TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L1TermKernel");
            m_L1TermKernel.SetupExecution(m_L1TermKernel.MAX_THREADS);
            m_L1TermKernel.DynamicSharedMemory = m_L1TermKernel.BlockDimensions.x * sizeof(float);

            m_L2TermKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "L2TermKernel");
            m_L2TermKernel.SetupExecution(m_L2TermKernel.MAX_THREADS);
            m_L2TermKernel.DynamicSharedMemory = m_L2TermKernel.BlockDimensions.x * sizeof(float);
            
            m_gaussianRegularizationKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "GaussianRegularizationKernel");
            m_gaussianRegularizationKernel.SetupExecution(m_gaussianRegularizationKernel.MAX_THREADS);
            m_gaussianRegularizationKernel.DynamicSharedMemory = m_gaussianRegularizationKernel.BlockDimensions.x * sizeof(float);
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

            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.RandomNormal.GetDevice(Owner), 0, 1);
            Owner.RandomNormal.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.Output.Count);

            // small HACK
            if (ResetPriorStats)
            {
                Owner.ResetPriorStats();
                ResetPriorStats = !ResetPriorStats;
            }
            // another snall HACK, store tr.rate the training rate if it is nonzero..
            if ((Owner.Parent as MyNeuralNetworkGroup).SGD.TrainingRate>0)
            {
                OriginalNeuralNetTrainingRate = (Owner.Parent as MyNeuralNetworkGroup).SGD.TrainingRate;
            }
            // If signal on, do generation, so overwrite the Input
            if (Owner.Generate.IsIncomingRised())
            {
                Owner.PriorGaussHiddenStatesMin.SafeCopyToHost();
                Owner.PriorGaussHiddenStatesMax.SafeCopyToHost();
                (Owner.Parent as MyNeuralNetworkGroup).SGD.TrainingRate = 0;
                for (int i = 0; i < Owner.Input.Count; i++)
                {
                    double diff = Owner.PriorGaussHiddenStatesMax.Host[i] - Owner.PriorGaussHiddenStatesMin.Host[i];
                    if (i < Owner.Input.Count / 2) Owner.Input.Host[i] = (float)(rnd.NextDouble() * diff + Owner.PriorGaussHiddenStatesMin.Host[i]);
                    else Owner.Input.Host[i] = 0.01f;
                }
                Owner.Input.SafeCopyToDevice();
            }
            else
            {
                (Owner.Parent as MyNeuralNetworkGroup).SGD.TrainingRate = OriginalNeuralNetTrainingRate;
            }

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

            // calculate Prior for generation
            Owner.Input.SafeCopyToHost();
            Owner.PriorGaussHiddenStatesMax.SafeCopyToHost();
            Owner.PriorGaussHiddenStatesMin.SafeCopyToHost();
            for (int i = 0; i < Owner.Input.Count; i++)
            {
                Owner.PriorGaussHiddenStatesMax.Host[i] = Math.Max(Owner.Input.Host[i], Owner.PriorGaussHiddenStatesMax.Host[i]);
                Owner.PriorGaussHiddenStatesMin.Host[i] = Math.Min(Owner.Input.Host[i], Owner.PriorGaussHiddenStatesMin.Host[i]);
            }
            Owner.PriorGaussHiddenStatesMax.SafeCopyToDevice();
            Owner.PriorGaussHiddenStatesMin.SafeCopyToDevice();

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

            m_gaussianRegularizationKernel.Run(
                means,
                sigmas,
                Owner.Input.Count,
                Owner.Regularization
            );
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
        // Properties
        [YAXSerializableField(DefaultValue = true)]
        [MyBrowsable, Category("Regularization")]
        public bool Regularize { get; set; }

        // Properties
        [YAXSerializableField(DefaultValue = 0.01f)]
        [MyBrowsable, Category("Regularization")]
        public float RegularizationCoefficient { get; set; }

        public MyGaussianBackDeltaTask() { }

        private MyCudaKernel m_samplingDeltaKernel;
        private MyCudaKernel m_regularizationDeltaKernel;

        private CUdeviceptr nullCUdeviceptr = new CUdeviceptr(0);

        public override void Init(int nGPU)
        {
            m_samplingDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "GaussianSamplingDeltaKernel");
            m_samplingDeltaKernel.SetupExecution(Owner.Neurons);

            m_regularizationDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\RegularizationTermKernels", "GaussianRegularizationDeltaKernel");
            m_regularizationDeltaKernel.SetConstantVariable<float>("RegularizationCoefficient", RegularizationCoefficient);
        }

        public override void Execute()
        {
            if (Owner.PreviousConnectedLayers.Count > 0)
            {
                // pointer to previous layer
                MyAbstractLayer previousLayer = Owner.PreviousConnectedLayers[0];

                if (previousLayer != null && previousLayer is MyAbstractWeightLayer)
                {
                    MyAbstractWeightLayer previousWeightLayer = previousLayer as MyAbstractWeightLayer;

                    // set locations for mean deltas
                    CUdeviceptr meanDeltas = previousLayer.Delta.GetDevicePtr(Owner, 0);
                    // set locations for sigma deltas
                    CUdeviceptr sigmaDeltas = previousLayer.Delta.GetDevicePtr(Owner, previousLayer.Delta.Count / 2);
                    
                    // reset delta
                    previousLayer.Delta.FillAll(0);

                    // determine input to previous layer
                    CUdeviceptr prevInputPtr = MyAbstractLayer.DetermineInput(previousLayer);

                    m_samplingDeltaKernel.Run(
                        Convert.ToInt32(Owner.UseSigmaConstant),
                        (int)previousLayer.ActivationFunction,
                        prevInputPtr,
                        meanDeltas,
                        sigmaDeltas,
                        Owner.Delta,
                        Owner.RandomNormal,
                        Owner.Neurons
                    );

                    if (Regularize)
                    {
                        int weightCount = previousWeightLayer.Weights.Count;
                        m_regularizationDeltaKernel.SetupExecution(weightCount);

                        m_regularizationDeltaKernel.Run(
                            Convert.ToInt32(Owner.UseSigmaConstant),
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
