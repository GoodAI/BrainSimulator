using GoodAI.Core;
using GoodAI.Core.Memory;
using ManagedCuda.BasicTypes;
using System;
using XmlFeedForwardNet.Networks;

namespace XmlFeedForwardNet.Layers
{
    public class MyPoolLayer : MyAbstractFBLayer
    {
        public enum MyPoolRule
        {
            MAX = 0,
            AVERAGE = 1
            // Add new rules
        }

        public MyMemoryBlock<float> ChosenInput { private set; get; }// Reference
        public SizeT ChosenInputOffset { private set; get; }
        public CUdeviceptr ChosenInputPtr { private set; get; }

        public uint Stride { private set; get; }
        public MyPoolRule PoolRule { private set; get; }

        private MyCudaKernel m_forwardKernel;
        private MyCudaKernel m_backwardKernel;

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyDeltaView(m_network, this, 0xFFDDDDFF);
        }*/



        public MyPoolLayer(MyAbstractFeedForwardNode network, uint stride, MyPoolRule poolRule)
            : base(network)
        {
            Stride = stride;
            PoolRule = poolRule;
        }


        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            m_output.Width = PreviousLayer.Output.Width / Stride;
            m_output.Height = PreviousLayer.Output.Height / Stride;
            m_output.Nb = PreviousLayer.Output.Nb;

            if (Output.Size == 0)
                throw new MyFeedForwardLayerException("PoolLayer: Output size cannot be 0");


        }



        public override void Initialize(Int32 nGPU)
        {
            // Create the kernels
            m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\PoolLayerKernel", "ForwardKernel");
            m_backwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\PoolLayerKernel", "BackwardKernel");

            base.Initialize(nGPU);

            // Forward
            ChosenInputPtr = ChosenInput.GetDevicePtr(m_network, ChosenInputOffset);

        }


        public override void Forward()
        {
            m_forwardKernel.SetupExecution(Output.Count);
            m_forwardKernel.Run((int)Stride,
                                (int)PoolRule,
                                OutputDataPtr,
                                PreviousLayer.OutputDataPtr,
                                ChosenInputPtr
                                );
        }





        public override void Backward()
        {

        }


        public override void BroadcastDelta()
        {
            m_backwardKernel.SetupExecution(PreviousLayer.Output.Count);
            m_backwardKernel.Run((int)PoolRule,
                                (int)Stride,
                                DeltaDataPtr,
                                (PreviousLayer as MyAbstractFBLayer).DeltaDataPtr,
                                ChosenInputPtr
                                );
        }

        public override void AllocateMemory()
        {
            base.AllocateMemory();

            // Store extra info in output memory block
            ChosenInput = m_network.OutputMemoryBlock;
            ChosenInputOffset = m_network.OutputMemoryBlock.Count;
            m_network.OutputMemoryBlock.Count += Output.Count;

        }
    }
}
