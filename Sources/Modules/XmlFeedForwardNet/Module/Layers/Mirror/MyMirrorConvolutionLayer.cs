using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using GoodAI.Core.Memory;
using GoodAI.Core.Observers;
using GoodAI.Core.Task;
using  XmlFeedForwardNet.Networks;
using GoodAI.Core;

namespace XmlFeedForwardNet.Layers.Mirror
{
    public class MyMirrorConvolutionLayer : MyAbstractWeightLayer
    {
        private MyConvolutionLayer m_originalLayer;

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_forwardBiasKernel;
        private MyCudaKernel m_backwardKernel;
        private MyCudaKernel m_weightKernel;
        private MyCudaKernel m_weightBiasesKernel;
        private MyCudaKernel m_setKernel;


        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyWeightView(m_network, this, 0xDDDDBBBB);
        }*/

        public MyMirrorConvolutionLayer(MyAbstractFeedForwardNode network, MyConvolutionLayer originalLayer, float[] initialWeights = null)
            : base(network)
        {
            m_originalLayer = originalLayer;

        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            m_output.Nb = m_originalLayer.PreviousLayer.Output.Nb;
            m_output.Width = m_originalLayer.PreviousLayer.Output.Width;
            m_output.Height = m_originalLayer.PreviousLayer.Output.Height;

            if (PreviousLayer.Output.Count != m_originalLayer.Output.Count)
                throw new MyFeedForwardLayerException("MirrorConvolutionLayer: input (" + PreviousLayer.Output.Size + "x" + PreviousLayer.Output.Nb + ") doesn't fit the output dimension of the referenced MyConvolutionLayer (" + m_originalLayer.Output.Size + "x" + m_originalLayer.Output.Nb + ")");


            //There are only biases since the synaptic weights are shared with the original layer
            m_bias = m_output;

        }

        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\MirrorConvolutionLayerKernel", "ForwardKernel");
            m_forwardBiasKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\MirrorConvolutionLayerKernel", "ForwardBiasKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\MirrorConvolutionLayerKernel", "BackwardKernel");
            m_weightBiasesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\MirrorConvolutionLayerKernel", "WeightBiasKernel");
            m_weightKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\MirrorConvolutionLayerKernel", "WeightKernel");

            base.Initialize(nGPU);

        }

        public override void Forward()
        {

            // Set the Output to zeros
            m_setKernel.SetupExecution(Output.Count);
            m_setKernel.Run(m_outputBlock, m_outputOffset, 0, Output.Count);


            // Sum all the outputs
            m_forwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_forwardKernel.Run(m_originalLayer.FeatureInfos,
                                PreviousLayer.OutputDataPtr,
                                OutputDataPtr,
                                m_originalLayer.WeightDataPtr,
                                m_originalLayer.XStride,
                                m_originalLayer.YStride
                                );

            // Add the bias
            m_forwardBiasKernel.SetupExecution(Output.Count);
            m_forwardBiasKernel.Run(OutputDataPtr,
                                    BiasDataPtr
                                    );
        }

        public override void Backward()
        {
            //Weight
            m_weightKernel.SetupExecution(PreviousLayer.Output.Count);
            m_weightKernel.Run(m_originalLayer.FeatureInfos,
                                PreviousLayer.OutputDataPtr,
                                DeltaDataPtr,
                                m_originalLayer.WeightChangeDataPtr,
                                m_originalLayer.XStride,
                                m_originalLayer.YStride
                                );

            // Biases
            m_weightBiasesKernel.SetupExecution(Bias.Count);
            m_weightBiasesKernel.Run(DeltaDataPtr,
                                    BiasChangeDataPtr
                                    );
        }

        public override void BroadcastDelta()
        {
            //Delta
            m_backwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_backwardKernel.Run(m_originalLayer.FeatureInfos,
                                OutputDataPtr,
                                DeltaDataPtr,
                                m_previousBackwardLayer.DeltaDataPtr,
                                m_originalLayer.WeightDataPtr,
                                m_originalLayer.XStride,
                                m_originalLayer.YStride
                                );
        }

        protected override void GenerateBiasFromRandom()
        {
            float biasInitialValue = 0f;
            m_setKernel.SetupExecution(Output.Count);
            m_setKernel.Run(m_weightBlock, m_weightOffset, biasInitialValue, Output.Count);
        }
    }
}
