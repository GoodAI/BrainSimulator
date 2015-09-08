using GoodAI.Core.Memory;
using ManagedCuda.BasicTypes;
using System;
using XmlFeedForwardNet.Networks;

namespace XmlFeedForwardNet.Layers
{
    public abstract class MyAbstractFBLayer : MyAbstractFLayer
    {
        public MyLayerDim Delta { get { return m_delta; } }
        public CUdeviceptr DeltaDataPtr { get; private set; }

        protected MyMemoryBlock<float> m_deltaBlock;
        protected SizeT m_deltaOffset;
        protected MyLayerDim m_delta;
        protected SizeT m_deltaDimGPUPtrOffset;

        protected MyAbstractFBLayer m_previousBackwardLayer;

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyDeltaView(m_network, this);
        }*/

        public MyAbstractFBLayer(MyAbstractFeedForwardNode network)
            : base(network)
        {
            m_delta.Depth = 1;
        }

        public override void Dimension(MyAbstractFLayer previousLayer)
        {
            base.Dimension(previousLayer);

            if (previousLayer is MyAbstractFBLayer)
                m_previousBackwardLayer = (previousLayer as MyAbstractFBLayer);
        }

        public override void AllocateMemory()
        {
            base.AllocateMemory();

            m_delta = m_output;

            m_deltaBlock = m_network.DeltasMemoryBlock;
            m_deltaOffset = m_network.DeltasMemoryBlock.Count;
            m_deltaDimGPUPtrOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;

            m_network.DeltasMemoryBlock.Count += m_delta.Count;
        }

        public override void Initialize(Int32 nGPU)
        {
            base.Initialize(nGPU);

            if (m_deltaBlock != null)
                m_delta.Ptr = m_deltaBlock.GetDevicePtr(m_network, m_deltaOffset);

            // Send the structures to GPU
            m_network.DataDimsMemoryBlock.Host[m_deltaDimGPUPtrOffset] = Delta;

            // Store the GPU pointers
            DeltaDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_deltaDimGPUPtrOffset);
        }

        public abstract void Backward();

        public abstract void BroadcastDelta();

        public virtual void Backward(float LearningRate, float LearningMomentum)
        {
            throw new NotImplementedException();
        }

    }
}
