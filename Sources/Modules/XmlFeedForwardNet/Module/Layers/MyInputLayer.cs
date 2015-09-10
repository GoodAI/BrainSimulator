using GoodAI.Core.Memory;
using ManagedCuda.BasicTypes;
using XmlFeedForwardNet.Networks;


namespace XmlFeedForwardNet.Layers
{
    public class MyInputLayer : MyAbstractFLayer
    {
        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyOutputView(m_network, this, 0xFFDDDDDD);
        }
         */

        protected MyMemoryBlock<float> m_inputBlock;
        protected SizeT m_inputOffset;
        protected SizeT m_nbSamplesPerStep;

        public MyInputLayer(MyAbstractFeedForwardNode network, MyMemoryBlock<float> input, SizeT offset, SizeT nb, SizeT width, SizeT height, SizeT nbSamplesPerStep)
            : base(network)
        {
            m_inputBlock = input;
            m_inputOffset = offset;

            m_output.Nb = nb;
            m_output.Width = width;
            m_output.Height = height;

            m_nbSamplesPerStep = nbSamplesPerStep;
        }

        // Replace the default function
        public override void AllocateMemory()
        {
            base.AllocateMemory();

            m_outputBlock = m_inputBlock;
            m_outputOffset = m_inputOffset;

            m_outputDimGPUOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;
        }

        public void SendInputSampleOffsetToGPU(uint offset)
        {
            if (m_outputBlock != null)
                m_output.Ptr = m_outputBlock.GetDevicePtr(m_network, m_outputOffset + offset * Output.Count);

            m_network.DataDimsMemoryBlock.Host[m_outputDimGPUOffset] = Output;

            m_network.DataDimsMemoryBlock.SafeCopyToDevice(m_outputDimGPUOffset, 1);
        }

        public void SetInputMemoryBlock(MyMemoryBlock<float> input, uint inputOffset = 0, uint sampleOffset = 0)
        {
            m_inputBlock = input;
            m_inputOffset = inputOffset;
            m_outputBlock = m_inputBlock;
            m_outputOffset = m_inputOffset;
            SendInputSampleOffsetToGPU(sampleOffset);
        }

        public override void Forward()
        {
        }
    }
}
