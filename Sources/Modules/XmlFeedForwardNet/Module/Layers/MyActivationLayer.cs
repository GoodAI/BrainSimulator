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
    public class MyActivationLayer : MyAbstractFBLayer
    {
        public MyActivationFunction ActivationFunction { get; private set; }


        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_backwardKernel;
        private MyCudaKernel m_setKernel;

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyDeltaView(m_network, this, 0xFFFFDDFF);
        }*/

        public enum MyActivationFunction
        {
            NO_ACTIVATION = 0,
            LOGISTIC = 1,
            RELU = 2,
            TANH = 3,
        }

        public MyActivationLayer(MyAbstractFeedForwardNode network, MyActivationFunction activationFunction = MyActivationFunction.NO_ACTIVATION)
            : base(network)
        {
            ActivationFunction = activationFunction;
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            // Just a clone from the previous layer (inplace computation)
            m_output = m_previousBackwardLayer.Output;
        }

        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\ActivationLayerKernel", "ForwardKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\ActivationLayerKernel", "BackwardKernel");

            base.Initialize(nGPU);
        }

        public override void Forward()
        {
            m_forwardKernel.SetupExecution(Output.Count);
            m_forwardKernel.Run((int)ActivationFunction,
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
            m_backwardKernel.Run((int)ActivationFunction,
                                OutputDataPtr,
                                DeltaDataPtr,
                                m_previousBackwardLayer.DeltaDataPtr
                                );
        }
    }
}
