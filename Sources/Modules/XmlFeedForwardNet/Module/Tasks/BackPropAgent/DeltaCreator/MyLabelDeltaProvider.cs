using GoodAI.Core;
using GoodAI.Core.Memory;
using ManagedCuda.BasicTypes;
using System;
using XmlFeedForwardNet.Networks;

namespace  XmlFeedForwardNet.Tasks.BackPropAgent.DeltaCreator
{
    public class MyLabelDeltaProvider : MyDeltaProvider
    {
        public MyMemoryBlock<float> LabelInput { get; set; }

        private MyCudaKernel m_combineKernel;
        private MyCudaKernel m_energyKernel;

        public MyLabelDeltaProvider(MyAbstractFeedForwardNode network, int nGPU)
            : base(network)
        {
            m_combineKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
            m_energyKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\EnergyKernel");
        }

        public CUdeviceptr CurrentSampleLabelInputPtr
        {
            get
            {
                if (LabelInput != null && m_network.LastLayer != null)
                    return LabelInput.GetDevicePtr(m_network, (int)(m_network.LastLayer.Output.Count * m_network.m_currentSamplePosition));
                else
                    return new CUdeviceptr(0);
            }
        }

        public override void Execute()
        {
            if (m_network.UseLabelsAsDeltas)
            {
                //copy deltas provided in Labels memory block to deltas of last layer
                m_network.m_copyKernel.SetupExecution(m_network.LastLayer.Output.Count);
                m_network.m_copyKernel.Run(CurrentSampleLabelInputPtr, 0, m_network.LastLayer.Delta.Ptr, 0, m_network.LastLayer.Output.Count);

                // energy is not computed in this case
            }
            else
            {
                //  Compare the last layer output with the label layer
                m_combineKernel.SetupExecution(m_network.LastLayer.Output.Count);
                m_combineKernel.Run(
                            m_network.LastLayer.Output.Ptr, // Network assumed
                            CurrentSampleLabelInputPtr, // Reference
                            m_network.LastLayer.Delta.Ptr,
                            1, // Substraction
                            m_network.LastLayer.Output.Count);

                // Compute the current energy of the network
                SaveEnergy();
            }
        }

        protected void SaveEnergy()
        {
            m_energyKernel.SetupExecution(1);
            m_energyKernel.Run(
                    Math.Min(m_network.SamplesProcessed, m_network.EnergySamplesCount),
                    m_network.LastLayer.Output.Ptr,
                    CurrentSampleLabelInputPtr,
                    m_network.LastLayer.Output.Count,
                    m_network.GetCurrentSampleEnergySlot(),
                    m_network.EnergySamples.GetDevicePtr(m_network),
                    m_network.Energy.GetDevicePtr(m_network)
                    );
        }
    }
}
