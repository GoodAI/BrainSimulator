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
using GoodAI.Core.Utils;
using System.Runtime.InteropServices;
using  XmlFeedForwardNet.Networks;

namespace XmlFeedForwardNet.Layers
{
    public abstract class MyAbstractWeightLayer : MyAbstractFBLayer
    {
        public MyLayerDim Weight { get { return m_weight; } }
        public MyLayerDim WeightChange { get { return m_weightChange; } }
        public MyLayerDim Bias { get { return m_bias; } }
        public MyLayerDim BiasChange { get { return m_biasChange; } }
        public MyLayerDim LastWeightDelta { get { return m_weight; } }
        public MyLayerDim StoredOutput { get { return m_bias; } }

        public CUdeviceptr WeightChangeDataPtr { get; private set; }
        public CUdeviceptr BiasChangeDataPtr { get; private set; }
        public CUdeviceptr LastWeightDeltaDataPtr { get; private set; }

        protected MyMemoryBlock<float> m_weightBlock;
        protected MyMemoryBlock<float> m_weightChangeBlock;
        protected MyMemoryBlock<float> m_biasBlock;
        protected MyMemoryBlock<float> m_biasChangeBlock;
        protected MyMemoryBlock<float> m_lastWeightDeltaBlock;
        protected MyMemoryBlock<float> m_storedOutputBlock;

        protected SizeT m_weightOffset;
        protected SizeT m_weightChangeOffset;
        protected SizeT m_biasOffset;
        protected SizeT m_biasChangeOffset;
        protected SizeT m_lastWeightDeltaOffset;
        protected SizeT m_storedOutputOffset;

        protected SizeT m_weightDimGPUPtrOffset;
        protected SizeT m_weightChangeDimGPUPtrOffset;
        protected SizeT m_biasDimGPUPtrOffset;
        protected SizeT m_biasChangeDimGPUPtrOffset;
        protected SizeT m_lastWeightDeltaDimGPUPtrOffset;
        protected SizeT m_storedOutputDimGPUPtrOffset;

        protected MyLayerDim m_weight;
        protected MyLayerDim m_weightChange;
        protected MyLayerDim m_bias;
        protected MyLayerDim m_biasChange;
        protected MyLayerDim m_lastWeightDelta;
        protected MyLayerDim m_storedOutput;

        protected float[] m_initialWeight;
        protected float[] m_initialBias;

        protected virtual void GenerateBiasFromRandom() { }
        protected virtual void GenerateWeightFromRandom() { }

        /* 
         * Observers not implemented
         * 
        public override MyOutputView CreateView()
        {
            throw new NotImplementedException();
            //return new MyWeightView(m_network, this);
        }*/

        public MyAbstractWeightLayer(MyAbstractFeedForwardNode network)
            : base(network)
        {
            m_weight.Depth = 1;
            m_weightChange.Depth = 1;
            m_bias.Depth = 1;
            m_biasChange.Depth = 1;
        }

        public override void AllocateMemory()
        {
            base.AllocateMemory();

            m_weightChange = m_weight;
            m_biasChange = m_bias;

            m_weightBlock = m_network.WeightsMemoryBlock;
            m_weightOffset = m_network.WeightsMemoryBlock.Count;
            m_network.WeightsMemoryBlock.Count += m_weight.Count;
            m_weightDimGPUPtrOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;

            m_weightChangeBlock = m_network.WeightChangesMemoryBlock;
            m_weightChangeOffset = m_network.WeightChangesMemoryBlock.Count;
            m_network.WeightChangesMemoryBlock.Count += m_weightChange.Count;
            m_weightChangeDimGPUPtrOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;

            m_biasBlock = m_network.WeightsMemoryBlock;
            m_biasOffset = m_network.WeightsMemoryBlock.Count;
            m_network.WeightsMemoryBlock.Count += m_bias.Count;
            m_biasDimGPUPtrOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;

            m_biasChangeBlock = m_network.WeightChangesMemoryBlock;
            m_biasChangeOffset = m_network.WeightChangesMemoryBlock.Count;
            m_network.WeightChangesMemoryBlock.Count += m_biasChange.Count;
            m_biasChangeDimGPUPtrOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;

            m_lastWeightDeltaBlock = m_network.WeightChangesMemoryBlock;
            m_lastWeightDeltaOffset = m_network.WeightChangesMemoryBlock.Count;
            m_network.WeightChangesMemoryBlock.Count += m_lastWeightDelta.Count;
            m_lastWeightDeltaDimGPUPtrOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;

            m_storedOutputBlock = m_network.WeightChangesMemoryBlock;
            m_storedOutputOffset = m_network.WeightChangesMemoryBlock.Count;
            m_network.WeightChangesMemoryBlock.Count += m_storedOutput.Count;
            m_storedOutputDimGPUPtrOffset = m_network.DataDimsMemoryBlock.Count;
            m_network.DataDimsMemoryBlock.Count++;
        }

        public override void Initialize(Int32 nGPU)
        {
            base.Initialize(nGPU);

            // Set WeightChange and BiasChange dimensions according to respective Weight and Bias

            if (m_weightBlock != null)
            {
                m_weight.Ptr = m_weightBlock.GetDevicePtr(m_network, m_weightOffset);
                m_weightChange.Ptr = m_weightChangeBlock.GetDevicePtr(m_network, m_weightChangeOffset);
            }
            if (m_biasBlock != null)
            {
                m_bias.Ptr = m_biasBlock.GetDevicePtr(m_network, m_biasOffset);
                m_biasChange.Ptr = m_biasChangeBlock.GetDevicePtr(m_network, m_biasChangeOffset);
            }

            // Send the structures to GPU
            m_network.DataDimsMemoryBlock.Host[m_weightDimGPUPtrOffset] = Weight;
            m_network.DataDimsMemoryBlock.Host[m_weightChangeDimGPUPtrOffset] = WeightChange;
            m_network.DataDimsMemoryBlock.Host[m_biasDimGPUPtrOffset] = Bias;
            m_network.DataDimsMemoryBlock.Host[m_biasChangeDimGPUPtrOffset] = BiasChange;
            m_network.DataDimsMemoryBlock.Host[m_lastWeightDeltaDimGPUPtrOffset] = LastWeightDelta;
            m_network.DataDimsMemoryBlock.Host[m_storedOutputDimGPUPtrOffset] = StoredOutput;

            // Store the GPU pointers
            WeightDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_weightDimGPUPtrOffset);
            WeightChangeDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_weightChangeDimGPUPtrOffset);
            BiasDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_biasDimGPUPtrOffset);
            BiasChangeDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_biasChangeDimGPUPtrOffset);
            LastWeightDeltaDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_lastWeightDeltaDimGPUPtrOffset);
            StoredOutputDataPtr = m_network.DataDimsMemoryBlock.GetDevicePtr(m_network, (int)m_storedOutputDimGPUPtrOffset);

            // Generate initial weights
            GenerateWeights();
        }

        private void GenerateWeights()
        {
            if (Weight.Count > 0)
            {
                if (m_initialWeight != null)
                    GenerateWeightFromInitialWeights();
                else
                    GenerateWeightFromRandom();
            }

            if (Bias.Count > 0)
            {
                if (m_initialBias != null)
                    GenerateBiasFromInitialWeights();
                else
                    GenerateBiasFromRandom();
            }
        }

        private void GenerateWeightFromInitialWeights()
        {
            m_weightBlock.SafeCopyToHost();
            for (int i = 0; i < m_initialWeight.Length; i++)
                m_weightBlock.Host[m_weightOffset + i] = m_initialWeight[i];
            m_weightBlock.SafeCopyToDevice();
        }

        private void GenerateBiasFromInitialWeights()
        {
            m_biasBlock.SafeCopyToHost();
            for (int i = 0; i < m_initialBias.Length; i++)
                m_biasBlock.Host[m_biasOffset + i] = m_initialBias[i];
            m_biasBlock.SafeCopyToDevice();
        }
    }
}
