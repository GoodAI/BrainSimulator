using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using GoodAI.Core.Memory;
using GoodAI.Core.Observers;
using XmlFeedForwardNet.Networks;
using GoodAI.Core.Task;
using GoodAI.Core;

namespace XmlFeedForwardNet.Layers
{
    public class MySoftmaxLayer : MyAbstractFBLayer
    {

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_backwardKernel;

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyDeltaView(m_network, this, 0xFFFFFFDD);
        }*/

        public MySoftmaxLayer(MyAbstractFeedForwardNode network)
            : base(network)
        {
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            m_output.Nb = PreviousLayer.Output.Nb;
            m_output.Width = PreviousLayer.Output.Width;
            m_output.Height = PreviousLayer.Output.Height;


        }

        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\SoftmaxLayerKernel", "ForwardKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\SoftmaxLayerKernel", "BackwardKernel");

            base.Initialize(nGPU);
        }

        public override void Forward()
        {
            m_forwardKernel.SetupExecution(Output.Count);
            m_forwardKernel.Run(OutputDataPtr, PreviousLayer.OutputDataPtr);
        }


        public override void Backward()
        {
        }



        public override void BroadcastDelta()
        {
            //m_backwardKernel.Run(DeltaDataPtr, m_previousBackwardLayer.DeltaDataPtr); // myhack
            m_backwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_backwardKernel.Run(DeltaDataPtr, m_previousBackwardLayer.DeltaDataPtr);   // myhack
        }
    }
}

