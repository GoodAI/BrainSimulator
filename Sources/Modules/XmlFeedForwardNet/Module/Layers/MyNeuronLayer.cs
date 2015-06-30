using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using System.Runtime.InteropServices;
using XmlFeedForwardNet.Networks;
using GoodAI.Core.Task;
using GoodAI.Core;

namespace XmlFeedForwardNet.Layers
{
    public class MyNeuronLayer : MyAbstractWeightLayer
    {
        private uint m_neuronsCount;

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_backwardKernel;
        private MyCudaKernel m_weightKernel;
        private MyCudaKernel m_biasKernel;
        private MyCudaKernel m_setKernel;
        private MyCudaKernel m_backpropKernel;

        private MyCudaKernel m_RBMForwardKernel;
        private MyCudaKernel m_RBMForwardAndStoreKernel;
        private MyCudaKernel m_RBMBackwardKernel;
        private MyCudaKernel m_RBMSampleKernel;
        private MyCudaKernel m_RBMUpdateWeightKernel;
        private MyCudaKernel m_RBMUpdateBiasKernel;
        private MyCudaKernel m_RBMObserverKernel;

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyWeightView(m_network, this, 0xFFDDFFFF);
        }*/

        public enum MyActivationFunction
        {
            NO_ACTIVATION = 0,
            LOGISTIC = 1,
            RELU = 2,
            TANH = 3,
        }


        public MyNeuronLayer(MyAbstractFeedForwardNode network, uint neuronsCount,
                                float[] initialWeights = null, float[] initialBias = null)
            : base(network)
        {
            m_neuronsCount = neuronsCount;

            m_output.Nb = m_neuronsCount;
            m_output.Width = 1;
            m_output.Height = 1;

            m_initialWeight = initialWeights;
            m_initialBias = initialBias;
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            // Set the weights
            m_weight.Nb = m_neuronsCount;
            m_weight.Width = PreviousLayer.Output.Width;
            m_weight.Height = PreviousLayer.Output.Height;
            m_weight.Depth = PreviousLayer.Output.Nb;

            m_bias.Nb = m_neuronsCount;
            m_bias.Width = 1;
            m_bias.Height = 1;

        }


        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\NeuronLayerKernel", "ForwardKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\NeuronLayerKernel", "BackwardKernel");
            m_weightKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\NeuronLayerKernel", "WeightKernel");
            m_biasKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\NeuronLayerKernel", "BiasKernel");
            m_backpropKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\NeuronLayerKernel", "BackpropKernel");

            m_RBMForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "ForwardKernel");
            m_RBMForwardAndStoreKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "ForwardAndStoreKernel");
            m_RBMBackwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "BackwardKernel");
            m_RBMSampleKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "SampleKernel");
            m_RBMUpdateWeightKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "UpdateWeightKernel");
            m_RBMUpdateBiasKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "UpdateBiasKernel");
            m_RBMObserverKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "ObserverKernel");

            base.Initialize(nGPU);
        }

        public override void Forward()
        {
            m_forwardKernel.SetupExecution(Output.Nb);
            m_forwardKernel.Run(OutputDataPtr,
                                PreviousLayer.OutputDataPtr,
                                WeightDataPtr,
                                BiasDataPtr
                                );
        }


        public override void Backward()
        {
            //Weights
            m_weightKernel.SetupExecution(Weight.Count);
            m_weightKernel.Run(PreviousLayer.OutputDataPtr,
                    DeltaDataPtr,
                    WeightChangeDataPtr
                );
            m_biasKernel.SetupExecution(Bias.Count);
            m_biasKernel.Run(DeltaDataPtr, BiasChangeDataPtr);
        }

        public override void Backward(float LearningRate, float LearningMomentum)
        {
            m_backpropKernel.SetupExecution(Weight.Count);
            m_backpropKernel.Run(
                    PreviousLayer.OutputDataPtr,
                    DeltaDataPtr,
                    WeightDataPtr,
                    LastWeightDeltaDataPtr,
                    LearningRate,
                    LearningMomentum,
                    Weight.Count
                );

            m_biasKernel.SetupExecution(Bias.Count);
            m_biasKernel.Run(DeltaDataPtr, BiasChangeDataPtr);
        }

        public override void BroadcastDelta()
        {
            m_backwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_backwardKernel.Run(DeltaDataPtr,
                                WeightDataPtr,
                                m_previousBackwardLayer.DeltaDataPtr
                                );
        }


        public override void RBMForward(MyAbstractFLayer previousLayer)
        {
            GenerateRBMRandom();

            m_RBMForwardKernel.SetupExecution(Output.Count);
            m_RBMForwardKernel.Run(previousLayer.OutputDataPtr, OutputDataPtr, WeightDataPtr, BiasDataPtr, m_network.RBMRandom.GetDevicePtr(m_network));
        }

        public override void RBMForwardAndStore(MyAbstractFLayer previousLayer)
        {
            GenerateRBMRandom();

            m_RBMForwardAndStoreKernel.SetupExecution(Output.Count);
            m_RBMForwardAndStoreKernel.Run(previousLayer.OutputDataPtr, OutputDataPtr, WeightDataPtr, BiasDataPtr, StoredOutputDataPtr, m_network.RBMRandom.GetDevicePtr(m_network));
        }

        public override void RBMBackward(MyAbstractFLayer nextLayer)
        {
            GenerateRBMRandom();

            m_RBMBackwardKernel.SetupExecution(Output.Count);
            m_RBMBackwardKernel.Run(nextLayer.OutputDataPtr, OutputDataPtr, nextLayer.WeightDataPtr, BiasDataPtr, 1, m_network.RBMRandom.GetDevicePtr(m_network));
        }

        public override void RBMSamplePositive(MyAbstractFLayer previousLayer)
        {
            m_RBMSampleKernel.SetupExecution(Weight.Count);
            //m_RBMSampleKernel.Run(previousLayer.OutputDataPtr, OutputDataPtr, RBMPositiveDataPtr, Weight.Count);
            m_RBMSampleKernel.Run(previousLayer.OutputDataPtr, OutputDataPtr, m_network.RBMPositiveMemoryBlock.GetDevicePtr(m_network), Weight.Count);
        }

        public override void RBMUpdate(MyAbstractFLayer previousLayer, float LearningRate, float Momentum, float WeightDecay)
        {

            m_RBMUpdateWeightKernel.SetupExecution(Weight.Count);
            m_RBMUpdateWeightKernel.Run(m_network.RBMPositiveMemoryBlock.GetDevicePtr(m_network), previousLayer.OutputDataPtr, OutputDataPtr, WeightDataPtr, m_network.RBMWeightMomentum.GetDevicePtr(m_network), LearningRate, Momentum, WeightDecay, m_network.Energy.GetDevicePtr(m_network), 1);

            // update biases of this (= hidden) layer
            m_RBMUpdateBiasKernel.SetupExecution(Output.Count);
            m_RBMUpdateBiasKernel.Run(StoredOutputDataPtr, OutputDataPtr, BiasDataPtr, m_network.RBMBiasMomentum2.GetDevicePtr(m_network), m_network.Energy.GetDevicePtr(m_network), LearningRate, Momentum, WeightDecay, 0);

            // update biases of previous (= visible) layer
            if (!(previousLayer is MyInputLayer))
            {
                m_RBMUpdateBiasKernel.SetupExecution(previousLayer.Output.Count);
                m_RBMUpdateBiasKernel.Run(previousLayer.StoredOutputDataPtr, previousLayer.OutputDataPtr, previousLayer.BiasDataPtr, m_network.RBMBiasMomentum1.GetDevicePtr(m_network), m_network.Energy.GetDevicePtr(m_network), LearningRate, Momentum, WeightDecay, 0);
            }

        }

        public void UpdateObserver(MyAbstractFLayer previousLayer)
        {
            m_RBMObserverKernel.SetupExecution(previousLayer.Output.Count);
            m_RBMObserverKernel.Run(previousLayer.OutputDataPtr, m_network.RBMObserver.GetDevicePtr(m_network));
        }

        public void UpdateObserver()
        {
            m_RBMObserverKernel.SetupExecution(Output.Count);
            m_RBMObserverKernel.Run(OutputDataPtr, m_network.RBMObserver2.GetDevicePtr(m_network));
        }


        /**************************
         * 
         *         WEIGHTS
         * 
         *************************/

        protected override void GenerateWeightFromRandom()
        {
            // Choose an appropriate StdDev
            // Trick found in The ConvNetJs project sources (file convnet_vol.js)
            // Allows to keep the same variance (=1) on every neuron
            float stdDev = (float)Math.Sqrt(1f / (int)(PreviousLayer.Output.Count + 1));

            // CUDA needs a even number of generated numbers
            int nbWeightsToGenerate = Weight.Count;
            if (nbWeightsToGenerate % 2 != 0)
                nbWeightsToGenerate = nbWeightsToGenerate + 1;

            MyKernelFactory.Instance.GetRandDevice(m_network).GenerateNormal32(Weight.Ptr, nbWeightsToGenerate, 0, stdDev);
        }

        protected override void GenerateBiasFromRandom()
        {
            // Set the bias to positive value
            float biasInitialValue = 0;
            m_setKernel.SetupExecution(Output.Count);
            m_setKernel.Run(Bias.Ptr, 0, biasInitialValue, Output.Count);
        }

        private void GenerateRBMRandom()
        {
            MyKernelFactory.Instance.GetRandDevice(m_network).GenerateUniform32(m_network.RBMRandom.GetDevicePtr(m_network), Output.Count);

            // observer (and/or program?) stops working when using nonrandom values for activations:
            //MyKernelFactory.Instance.GetRandDevice(m_network).GenerateNormal32(m_network.RBMRandom.GetDevicePtr(m_network), Output.Count, 0.51f, 0f);
            //m_network.RBMRandom.Fill(0.5f);
        }
    }
}
