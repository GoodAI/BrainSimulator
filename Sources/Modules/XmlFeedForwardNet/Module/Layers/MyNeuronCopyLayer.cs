using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Observers;
using System.Runtime.InteropServices;
using  XmlFeedForwardNet.Networks;
using BrainSimulator.Task;
using BrainSimulator;

namespace XmlFeedForwardNet.Layers
{
    /// <summary>
    /// This layer has no incoming weights and it copies inputs to its output when forwarding.
    /// This is an extension/copy of the input layer.
    /// INPUT: O   O   O
    ///        |   |   |
    /// COPYL: O   O   O
    ///         .......
    /// HIDDL: O O O O O
    /// </summary>
    public class MyNeuronCopyLayer : MyAbstractWeightLayer
    {
        private uint m_neuronsCount;

        // what to do with biases here? RBM uses them, Backprop does not
        private MyCudaKernel m_setKernel;

        private MyCudaKernel m_RBMBackwardKernel;
        private MyCudaKernel m_RBMObserverKernel;

        private MyCudaKernel m_ForwardKernel;

        private MyCudaKernel m_ForwardAndStoreKernel;

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


        public MyNeuronCopyLayer(MyAbstractFeedForwardNode network, uint neuronsCount,
                                float[] initialWeights = null, float[] initialBias = null)
            : base(network)
        {
            m_neuronsCount = neuronsCount;

            m_output.Nb = m_neuronsCount;
            m_output.Width = 1;
            m_output.Height = 1;

            m_initialWeight = null;

            // init biases to 0
            m_initialBias = new float[m_output.Count];
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            // Set the weights
            m_weight.Nb = 0;
            m_weight.Width = 0;
            m_weight.Height = 0;
            m_weight.Depth = 0;

            m_bias.Nb = m_neuronsCount;
            m_bias.Width = 1;
            m_bias.Height = 1;

        }


        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");

            m_RBMBackwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "BackwardKernel");
            m_RBMObserverKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "ObserverKernel");

            m_ForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "NeuronCopyForwardKernel");
            m_ForwardAndStoreKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\RBMKernel", "NeuronCopyForwardAndStoreKernel");

            base.Initialize(nGPU);
        }

        public override void Forward()
        {
            MyKernelFactory.Instance.GetRandDevice(m_network).GenerateUniform32(m_network.RBMRandom.GetDevicePtr(m_network), Output.Count);

            m_ForwardKernel.SetupExecution(Output.Count);
            m_ForwardKernel.Run(PreviousLayer.OutputDataPtr,
                                OutputDataPtr,
                                BiasDataPtr,
                                m_network.RBMRandom.GetDevicePtr(m_network)
                                );
        }

        public override void Backward()
        {
        }

        public override void BroadcastDelta()
        {
        }


        public override void RBMForward(MyAbstractFLayer previousLayer)
        {
            GenerateRBMRandom();

            m_ForwardKernel.SetupExecution(Output.Count);
            m_ForwardKernel.Run(previousLayer.OutputDataPtr, OutputDataPtr, BiasDataPtr, m_network.RBMRandom.GetDevicePtr(m_network));
        }

        public override void RBMBackward(MyAbstractFLayer nextLayer)
        {
            GenerateRBMRandom();

            m_RBMBackwardKernel.SetupExecution(Output.Count);
            m_RBMBackwardKernel.Run(nextLayer.OutputDataPtr, OutputDataPtr, nextLayer.WeightDataPtr, BiasDataPtr, 1, m_network.RBMRandom.GetDevicePtr(m_network));
        }

        public override void RBMForwardAndStore(MyAbstractFLayer previousLayer)
        {
            GenerateRBMRandom();

            m_ForwardAndStoreKernel.SetupExecution(this.Output.Count);
            m_ForwardAndStoreKernel.Run(previousLayer.OutputDataPtr, OutputDataPtr, BiasDataPtr, StoredOutputDataPtr, m_network.RBMRandom.GetDevicePtr(m_network));
        }


        // TODO fix
        //public void UpdateObserver(MyAbstractFLayer previousLayer)
        public void UpdateObserver()
        {
            m_RBMObserverKernel.SetupExecution(Output.Count);
            m_RBMObserverKernel.Run(OutputDataPtr, m_network.RBMObserver.GetDevicePtr(m_network));
        }

        /**************************
         * 
         *         WEIGHTS
         * 
         *************************/

        protected override void GenerateWeightFromRandom()
        {
            // Neuron copy layer has no weights (or has weight 1 for each w(i,j), i == j
            // and 0 for w(i,j), i != j
        }

        protected override void GenerateBiasFromRandom()
        {
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
