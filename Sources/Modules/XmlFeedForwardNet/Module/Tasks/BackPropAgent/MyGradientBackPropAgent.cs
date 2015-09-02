using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using XmlFeedForwardNet.Layers;
using XmlFeedForwardNet.Networks;
using XmlFeedForwardNet.Tasks.BackPropAgent.DeltaCreator;


namespace XmlFeedForwardNet.Tasks.BackPropAgent
{
    public class MyGradientBackPropAgent : MyBackPropAgent
    {
        private MyCudaKernel m_updateWeightKernel;
        public MyLabelDeltaProvider DeltaProvider;

        public MyGradientBackPropAgent(MyAbstractFeedForwardNode network, int nGPU, MyMemoryBlock<float> labelInput)
            : base(network)
        {
            m_updateWeightKernel = MyKernelFactory.Instance.Kernel(nGPU, @"XmlFeedForwardNet\UpdateWeightKernel");
            DeltaProvider = new MyLabelDeltaProvider(m_network, nGPU);
            DeltaProvider.LabelInput = labelInput;
        }


        public override void Execute(uint trainingStep)
        {
            // May skip this task if the Learning period is over
            if (LearningDuration != 0 && trainingStep >= LearningDuration)
            {
                if (LearningDuration == trainingStep)
                    MyLog.INFO.WriteLine("Backpropagation stopped after " + LearningDuration + " steps.");
                return;
            }



            if (false)
            // TODO - don't use batches at all when using online learning
            // would need reimplementation of all layers, currently only NeuronLayer supports it
            //if (LearningBatchSize <= 1)
            {
                // Create the delta for the last layer
                DeltaProvider.Execute();

                // Propagate delta through all layers
                for (int layerId = m_network.Layers.Count - 1; layerId >= 0; layerId--)
                {
                    MyAbstractFBLayer layer = m_network.Layers[layerId];
                    if (layerId > 0)
                        layer.BroadcastDelta();
                }

                // BackPropagate all the layers (= infer weight changes and apply them immediately)
                for (int layerId = m_network.Layers.Count - 1; layerId >= 0; layerId--)
                {
                    MyAbstractFBLayer layer = m_network.Layers[layerId];
                    layer.Backward(LearningRate, LearningMomentum);
                }


            }



            // batch learning
            else
            {
                // Create the delta for the last layer
                DeltaProvider.Execute();

                // Propagate delta through all layers
                for (int layerId = m_network.Layers.Count - 1; layerId >= 0; layerId--)
                {
                    MyAbstractFBLayer layer = m_network.Layers[layerId];
                    if (layerId > 0)
                        layer.BroadcastDelta();
                }

                // BackPropagate all the layers (= infer weight changes)
                for (int layerId = m_network.Layers.Count - 1; layerId >= 0; layerId--)
                {
                    MyAbstractFBLayer layer = m_network.Layers[layerId];
                    layer.Backward();
                }

                // Update the weights (= apply the weight changes) if the batch is full
                if (m_network.WeightsMemoryBlock.Count > 0 && m_network.SamplesProcessed % LearningBatchSize == 0)
                {
                    m_updateWeightKernel.SetupExecution(m_network.WeightChangesMemoryBlock.Count);
                    m_updateWeightKernel.Run(
                                (uint)m_network.WeightChangesMemoryBlock.Count,
                                LearningRate,
                                LearningMomentum,
                                m_network.WeightsMemoryBlock.GetDevicePtr(m_network),
                                m_network.WeightChangesMemoryBlock.GetDevicePtr(m_network),
                                m_network.LastWeightDeltasMemoryBlock.GetDevicePtr(m_network)
                        );
                }


            }


        }
    }
}
