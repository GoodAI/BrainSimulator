using System;
using BrainSimulator.Memory;
using ManagedCuda.BasicTypes;
using  XmlFeedForwardNet.Networks;

namespace XmlFeedForwardNet.Layers
{

    public abstract class MyAbstractFLayer
    {
        public MyAbstractFLayer PreviousLayer { get; protected set; }
        public MyAbstractFLayer NextLayer { get; set; }
        public MyLayerDim Output { get { return m_output; } }

        // GPU memory pointers
        public CUdeviceptr OutputDataPtr { get; protected set; }
        public CUdeviceptr WeightDataPtr { get; protected set; }
        public CUdeviceptr BiasDataPtr { get; protected set; }
        public CUdeviceptr StoredOutputDataPtr { get; protected set; }

        protected MyAbstractFeedForwardNode m_network; // Reference

        protected MyMemoryBlock<float> m_outputBlock;
        protected SizeT m_outputOffset;
        protected MyLayerDim m_output;
        protected SizeT m_outputDimGPUOffset;


        protected MyMemoryBlock<uint> m_extraBlock;
        protected SizeT m_extraOffset;
        protected SizeT m_extraSize;
        protected CUdeviceptr m_extraPtr;

        /* 
         * Observers not implemented
         * 
        public virtual MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyOutputView(m_network, this);
        }*/

        public MyAbstractFLayer(MyAbstractFeedForwardNode network)
        {
            m_output.Depth = 1;
            m_network = network;
            m_extraSize = 0;
        }

        public virtual void Dimension(MyAbstractFLayer previousLayer)
        {
            PreviousLayer = previousLayer;

            if (PreviousLayer != null)
            {
                if (PreviousLayer.Output.Size == 0)
                    throw new MyFeedForwardLayerException("AbstractFLayer: Input size is 0");

                if (PreviousLayer.Output.Width == 0)
                    throw new MyFeedForwardLayerException("AbstractFLayer: Input width is 0");

                if (PreviousLayer.Output.Width * PreviousLayer.Output.Height != PreviousLayer.Output.Size)
                    throw new MyFeedForwardLayerException("AbstractFLayer: Requires a rectangular input");
            }
        }

        public virtual void AllocateMemory()
        {
            // output
            m_outputBlock = m_network.OutputMemoryBlock;
            m_outputOffset = m_network.OutputMemoryBlock.Count;

            m_outputDimGPUOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;

            m_network.OutputMemoryBlock.Count += m_output.Count;

            // extra
            m_extraBlock = m_network.ExtraMemoryBlock;
            m_extraOffset = m_network.ExtraMemoryBlock.Count;

            m_network.ExtraMemoryBlock.Count += m_extraSize;
        }

        public virtual void Initialize(Int32 nGPU)
        {
            // output

            // Set the dimension CUDevicePtr
            if (m_outputBlock != null)
                m_output.Ptr = m_outputBlock.GetDevicePtr(m_network, m_outputOffset);

            m_network.DataDimsMemoryBlock.Host[m_outputDimGPUOffset] = Output;

            // Store the GPU pointers
            OutputDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_outputDimGPUOffset);


            // extra
            if (m_extraBlock != null)
                m_extraPtr = m_extraBlock.GetDevicePtr(m_network, m_extraOffset);


        }

        public abstract void Forward();

        public virtual void RBMForwardAndStore(MyAbstractFLayer previousLayer)
        {
            throw new NotImplementedException();
        }

        public virtual void RBMSamplePositive(MyAbstractFLayer previousLayer)
        {
            throw new NotImplementedException();
        }

        public virtual void RBMBackward(MyAbstractFLayer nextLayer)
        {
            throw new NotImplementedException();
        }

        public virtual void RBMForward(MyAbstractFLayer previousLayer)
        {
            throw new NotImplementedException();
        }

        public virtual void RBMUpdate(MyAbstractFLayer previousLayer, float LearningRate, float LearningMomentum, float WeightDecay)
        {
            throw new NotImplementedException();
        }
    }
}
