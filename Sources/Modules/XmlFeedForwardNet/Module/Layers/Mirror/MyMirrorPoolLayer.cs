using GoodAI.Core;
using System;
using XmlFeedForwardNet.Networks;

namespace XmlFeedForwardNet.Layers
{
    public class MyMirrorPoolLayer : MyAbstractFBLayer
    {
        private MyPoolLayer m_originalLayer;

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_backwardKernel;

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyDeltaView(m_network, this, 0xDDBBBBDD);
        }*/

        public MyMirrorPoolLayer(MyAbstractFeedForwardNode network, MyPoolLayer originalLayer)
            : base(network)
        {
            m_originalLayer = originalLayer;
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            if (PreviousLayer.Output.Count != m_originalLayer.Output.Count)
                throw new MyFeedForwardLayerException("MirrorPoolLayer: Input (" + PreviousLayer.Output.Size + "x" + PreviousLayer.Output.Nb + ") doesn't fit the output dimension of the referenced MyPoolLayer (" + m_originalLayer.Output.Size + "x" + m_originalLayer.Output.Nb + ")");

            // Set the output
            m_output.Nb = m_originalLayer.PreviousLayer.Output.Nb;
            m_output.Height = m_originalLayer.PreviousLayer.Output.Height;
            m_output.Width = m_originalLayer.PreviousLayer.Output.Width;

            if (Output.Size == 0)
                throw new MyFeedForwardLayerException("MirrorPoolLayer: Output size cannot be 0");
        }

        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\MirrorPoolLayerKernel", "ForwardKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\MirrorPoolLayerKernel", "BackwardKernel");

            base.Initialize(nGPU);

        }

        public override void Forward()
        {
            m_forwardKernel.SetupExecution(Output.Count);
            m_forwardKernel.Run((uint)m_originalLayer.PoolRule,
                                (uint)m_originalLayer.Stride,
                                OutputDataPtr,
                                PreviousLayer.OutputDataPtr
                                );
        }

        public override void Backward()
        {
        }

        public override void BroadcastDelta()
        {
            m_backwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_backwardKernel.Run((uint)m_originalLayer.PoolRule,
                                (uint)m_originalLayer.Stride,
                                DeltaDataPtr,
                                m_previousBackwardLayer.DeltaDataPtr
                                );
        }
    }
}
