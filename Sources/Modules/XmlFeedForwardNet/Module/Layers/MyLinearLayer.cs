using GoodAI.Core;
using System;
using XmlFeedForwardNet.Networks;

namespace XmlFeedForwardNet.Layers
{
    public class MyLinearLayer : MyAbstractWeightLayer
    {
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
            //return new MyWeightView(m_network, this, 0xFFD8BD99);
        }*/

        public MyLinearLayer(MyAbstractFeedForwardNode network,
                                float[] initialWeights = null, float[] initialBias = null)
            : base(network)
        {
            m_initialWeight = initialWeights;
            m_initialBias = initialBias;
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            // Just a clone from the previous layer
            m_output = m_previousBackwardLayer.Output;

            // Set the weights (1 weight + 1 bias per output)
            m_weight = m_output;
            m_bias = m_output;
        }

        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\LinearLayerKernel", "ForwardKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\LinearLayerKernel", "BackwardKernel");
            m_weightKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\LinearLayerKernel", "WeightKernel");
            m_biasKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\LinearLayerKernel", "BiasKernel");

            base.Initialize(nGPU);
        }

        public override void Forward()
        {
            m_forwardKernel.SetupExecution(Output.Count);
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
            m_biasKernel.Run(DeltaDataPtr,
                             BiasChangeDataPtr
                             );
        }

        public override void BroadcastDelta()
        {
            m_backwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_backwardKernel.Run(DeltaDataPtr,
                                 WeightDataPtr,
                                 m_previousBackwardLayer.DeltaDataPtr
                                 );
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
            //float stdDev = (float)Math.Sqrt(1f / (int)(PreviousLayer.Output.Count + 1));

            //// CUDA needs a even number of generated numbers
            //int nbWeightsToGenerate = Weight.Count;
            //if (nbWeightsToGenerate % 2 != 0)
            //    nbWeightsToGenerate = nbWeightsToGenerate + 1;
            //MyKernelFactory.Instance.GetRandDevice(m_network).GenerateNormal32(Weight.Ptr, nbWeightsToGenerate, 0, stdDev);

            m_setKernel.SetupExecution(Weight.Count);
            m_setKernel.Run(Weight.Ptr, 0, 1f, Weight.Count);
        }

        protected override void GenerateBiasFromRandom()
        {
            float biasInitialValue = 0;
            m_setKernel.SetupExecution(Bias.Count);
            m_setKernel.Run(Bias.Ptr, 0, biasInitialValue, Bias.Count);
        }
    }
}
