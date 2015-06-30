using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using XmlFeedForwardNet.Layers;
using ManagedCuda.BasicTypes;
using XmlFeedForwardNet.Tasks;
using XmlFeedForwardNet.Utils;
using GoodAI.Core;

namespace XmlFeedForwardNet.Networks
{



    public class MyAbstractFeedForwardNode : MyWorkingNode
    {
        public enum MyLearningMethod
        {
            GRADIENT_DESCENT
        }

        /****************************
        *        PARAMETERS
        ****************************/


        public uint InputWidth { get; set; }

        public uint InputHeight { get; set; }

        public uint InputsCount { get; set; }

        public uint ForwardSamplesPerStep { get; set; }

        public uint TrainingSamplesPerStep { get; set; }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tLearning"), Description("Provide your own deltas into the label input and use them for training.")]
        public bool UseLabelsAsDeltas { get; set; }

        [YAXSerializableField(DefaultValue = (uint)1), YAXSerializeAs("EnergyNbSamples")]
        protected uint m_energySamples = 1;
        [MyBrowsable, Category("\tLearning"), Description("Number of samples used to compute the energy. The higher the number, the smoother the curve.")]
        public uint EnergySamplesCount
        {
            get { return m_energySamples; }
            set { if (value > 0) m_energySamples = value; }
        }

        [YAXSerializableField(DefaultValue = MyLearningMethod.GRADIENT_DESCENT)]
        [MyBrowsable, Category("\tLearning"), Description("Learning method")]
        public MyLearningMethod LearningMethod { get; set; }

        /****************************
        *     PUBLIC PROPERTIES
        ****************************/

        public MyCudaKernel m_copyKernel;
        public MyCudaKernel m_setKernel;

        public MyMemoryBlock<float> OutputMemoryBlock { get; set; }
        public MyMemoryBlock<float> DeltasMemoryBlock { get; set; }
        public MyMemoryBlock<uint> ExtraMemoryBlock { get; set; }
        [MyPersistable]
        public MyMemoryBlock<float> WeightsMemoryBlock { get; set; }
        public MyMemoryBlock<float> WeightChangesMemoryBlock { get; set; }
        public MyMemoryBlock<float> LastWeightDeltasMemoryBlock { get; set; }
        public MyMemoryBlock<MyLayerDim> DataDimsMemoryBlock { get; set; }
        public MyMemoryBlock<float> MinMax { get; set; }

        public MyMemoryBlock<float> RBMPositiveMemoryBlock { get; set; }
        public MyMemoryBlock<float> StoredOutputBlock { get; set; }
        public MyMemoryBlock<float> RBMWeightMomentum { get; set; }
        public MyMemoryBlock<float> RBMBiasMomentum1 { get; set; }
        public MyMemoryBlock<float> RBMBiasMomentum2 { get; set; }
        public MyMemoryBlock<float> RBMObserver { get; set; }
        public MyMemoryBlock<float> RBMObserver2 { get; set; }
        public MyMemoryBlock<float> RBMRandom { get; set; }

        public MyMemoryBlock<float> EnergySamples { get; set; }
        public MyMemoryBlock<float> Energy { get; set; }

        public bool ParamsChanged { get; set; }
        public MyInputLayer InputLayer { get; set; }
        public List<MyAbstractFBLayer> Layers { get; private set; }
        public MyAbstractFBLayer LastLayer { get; private set; }

        public uint SamplesProcessed { get; set; }
        public uint m_currentSamplePosition;

        /****************************
        *          METHODS
        ****************************/

        public MyAbstractFeedForwardNode()
            : base()
        {
            EnergySamplesCount = 1;
            ParamsChanged = true;
            m_currentSamplePosition = 0;
        }

        public void AddLayer(MyAbstractFBLayer layer)
        {
            Layers.Add(layer);
        }

        protected virtual void Build()
        {
        }

        protected virtual void Allocate()
        {
            InputLayer.Dimension(null);
            InputLayer.AllocateMemory();

            int maxLayerLength = 0;
            int maxWeightCount = 0;
            for (int i = 0; i < Layers.Count; i++)
            {
                if (i == 0)
                    Layers[i].Dimension(InputLayer);
                else
                    Layers[i].Dimension(Layers[i - 1]);
                Layers[i].AllocateMemory();

                // Because CuRand need multiple of 8, we need to align it. Float size is 4, so we need a multiple of 2.
                if (WeightsMemoryBlock.Count % 2 != 0)
                    WeightsMemoryBlock.Count += 1;

                StoredOutputBlock.Count += Layers[i].Output.Nb;
                maxLayerLength = Math.Max(maxLayerLength, Layers[i].Output.Count);

            }

            RBMObserver.Count = InputLayer.Output.Count;
            RBMObserver2.Count = Layers[Layers.Count - 1].Output.Count;

            RBMBiasMomentum1.Count = maxLayerLength;
            RBMBiasMomentum2.Count = maxLayerLength;
            RBMRandom.Count = maxLayerLength;

            for (int i = 0; i < Layers.Count - 1; i++)
            { 
                Layers[i].NextLayer = Layers[i+1];
                maxWeightCount = Math.Max(maxWeightCount, Layers[i].Output.Count * Layers[i+1].Output.Count);
            }
            maxWeightCount = Math.Max(maxWeightCount, InputLayer.Output.Count * Layers[0].Output.Count);
            RBMPositiveMemoryBlock.Count = maxWeightCount;
            RBMWeightMomentum.Count = maxWeightCount;
            //RBMPositiveMemoryBlock.Count = 1;


        }

        protected virtual void updateInput()
        {
        }

        protected virtual void Dimension()
        {
            m_currentSamplePosition = 0;

            OutputMemoryBlock.Count = 0;
            DeltasMemoryBlock.Count = 0;
            WeightsMemoryBlock.Count = 0;
            WeightChangesMemoryBlock.Count = 0;
            ExtraMemoryBlock.Count = 0;
            DataDimsMemoryBlock.Count = 0;

            Energy.Count = 1;
            EnergySamples.Count = (int)EnergySamplesCount;


            Layers = new List<MyAbstractFBLayer>();

            Build();    // parse?

            Allocate();

            SetHints();

            // Define the output
            LastLayer = Layers[Layers.Count - 1];

            ParamsChanged = false;
        }

        protected virtual void SetHints()
        {
            OutputMemoryBlock.MinValueHint = -1;
            OutputMemoryBlock.MaxValueHint = +1;
            DeltasMemoryBlock.MinValueHint = -1;
            DeltasMemoryBlock.MaxValueHint = +1;
            WeightsMemoryBlock.MinValueHint = -1;
            WeightsMemoryBlock.MaxValueHint = +1;
            WeightChangesMemoryBlock.MinValueHint = -1;
            WeightChangesMemoryBlock.MaxValueHint = +1;
            LastWeightDeltasMemoryBlock.Count = WeightChangesMemoryBlock.Count;

            MinMax.Count = 2;

            ExtraMemoryBlock.ColumnHint = 20;
        }

        public void ResetSample()
        {
            m_currentSamplePosition = 0;
            InputLayer.SendInputSampleOffsetToGPU(m_currentSamplePosition);
        }

        public void NextSample()
        {
            m_currentSamplePosition++;
            InputLayer.SendInputSampleOffsetToGPU(m_currentSamplePosition);
        }

        /***********************
         *    NETWORK ENERGY
         * ********************/

        public virtual CUdeviceptr GetCurrentSampleEnergySlot()
        {
            return EnergySamples.GetDevicePtr(this, (int)((SamplesProcessed - 1) % EnergySamplesCount));
        }

        public override void UpdateMemoryBlocks()
        {
            updateInput();
            if (ParamsChanged)
            {
                try
                {
                    Dimension();
                }
                catch (MyFeedForwardLayerException e)
                {
                    ParamsChanged = false;
                    MyLog.ERROR.WriteLine(e.Message);
                }
                catch (MyXMLNetworkBuilder.MyXMLBuilderException e)
                {
                    ParamsChanged = false;
                    MyLog.ERROR.WriteLine(e.Message);
                }

            }
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(!ParamsChanged, this, "Some parameters have changed. Network needs to be rebuilt.");
        }
    }
}
