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
using  XmlFeedForwardNet.Networks;
using BrainSimulator.Task;
using BrainSimulator;

namespace XmlFeedForwardNet.Layers
{
    public class MyMirrorNeuronLayer : MyAbstractWeightLayer
    {
        private float[] m_initialWeights;

        private MyNeuronLayer m_originalLayer;

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_backwardKernel;
        private MyCudaKernel m_weightKernel;
        private MyCudaKernel m_biasKernel;
        private MyCudaKernel m_setKernel;

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyWeightView(m_network, this, 0xCCAACCCC);
        }*/


        public MyMirrorNeuronLayer(MyAbstractFeedForwardNode network, MyNeuronLayer originalLayer, float[] initialWeights = null)
            : base(network)
        {
            m_originalLayer = originalLayer;

            m_initialWeights = initialWeights;
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            if (PreviousLayer.Output.Count != m_originalLayer.Output.Count)
                throw new MyFeedForwardLayerException("MirrorNeuronLayer: input (" + PreviousLayer.Output.Size + "x" + PreviousLayer.Output.Nb + ") doesn't fit the output dimension of the referenced MyNeuronLayer (" + m_originalLayer.Output.Size + "x" + m_originalLayer.Output.Nb + ")");

            // Set the output
            m_output.Nb = m_originalLayer.PreviousLayer.Output.Nb;
            m_output.Height = m_originalLayer.PreviousLayer.Output.Height;
            m_output.Width = m_originalLayer.PreviousLayer.Output.Width;

            // There are only biases since the synaptic weights are shared with the original layer
            m_bias.Width = Output.Width;
            m_bias.Height = Output.Height;
            m_bias.Nb = Output.Nb;
        }

        public override void Initialize(Int32 nGPU)
        {

            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");

            // Use the same kernels as MyNeuronLayer
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"FeatureDetection\MirrorNeuronLayerKernel", "ForwardKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"FeatureDetection\MirrorNeuronLayerKernel", "BackwardKernel");
            m_weightKernel = MyKernelFactory.Instance.Kernel(nGPU, @"FeatureDetection\MirrorNeuronLayerKernel", "WeightKernel");
            m_biasKernel = MyKernelFactory.Instance.Kernel(nGPU, @"FeatureDetection\MirrorNeuronLayerKernel", "BiasKernel");

            base.Initialize(nGPU);
        }

        public override void Forward()
        {
            m_forwardKernel.SetupExecution(Output.Count);
            m_forwardKernel.Run(OutputDataPtr,
                                PreviousLayer.OutputDataPtr,
                                m_originalLayer.WeightDataPtr,
                                BiasDataPtr
                                );
        }

        public override void Backward()
        {
            //Weight
            m_weightKernel.SetupExecution(m_originalLayer.Weight.Count);
            m_weightKernel.Run(PreviousLayer.OutputDataPtr,
                               DeltaDataPtr,
                               m_originalLayer.WeightChangeDataPtr
                               );
            m_biasKernel.SetupExecution(Bias.Count);
            m_biasKernel.Run(DeltaDataPtr,
                             BiasChangeDataPtr
                             );
        }

        public override void BroadcastDelta()
        {
            //Delta
            m_backwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_backwardKernel.Run(DeltaDataPtr,
                                 m_originalLayer.WeightDataPtr,
                                 m_previousBackwardLayer.DeltaDataPtr
                                 );
        }

        /**************************
         * 
         *         WEIGHTS
         * 
         *************************/

        protected override void GenerateBiasFromRandom()
        {
            // Set the bias
            float biasInitialValue = 0f;
            m_setKernel.SetupExecution(Bias.Count);
            m_setKernel.Run(m_biasBlock, m_biasOffset, biasInitialValue, Bias.Count);
        }
    }
}
