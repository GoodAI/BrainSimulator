using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Utils;
using GoodAI.Core.Task;
using  XmlFeedForwardNet.Networks;
using XmlFeedForwardNet.Layers;
using XmlFeedForwardNet.Tasks;
using  XmlFeedForwardNet.Tasks.BackPropAgent;
using  XmlFeedForwardNet.Tasks.BackPropAgent.DeltaCreator;
using GoodAI.Core.Memory;


namespace  XmlFeedForwardNet.Tasks.BackPropAgent
{
    public class MyRBMAgent : MyBackPropAgent
    {
        public uint ContrastiveDivergenceParameter;

        /// <summary>
        /// Number of iterations per layer pair * (number of layer pairs (=number of hidden layers))
        /// </summary>
        private int totalSteps;
        private List<MyAbstractFBLayer> layers;

        public float WeightDecay { get; set; }

        public MyRBMAgent(MyAbstractFeedForwardNode network, int nGPU, MyMemoryBlock<float> labelInput, uint learningDuration)
            : base(network)
        {
            layers = new List<MyAbstractFBLayer>();
            foreach (MyAbstractFBLayer l in network.Layers)
            {
                MyNeuronLayer nl = l as MyNeuronLayer;
                if (nl != null)
                    layers.Add(nl);
                MyNeuronCopyLayer ncl = l as MyNeuronCopyLayer;
                if (ncl != null)
                    layers.Add(ncl);
            }
            totalSteps = (int)learningDuration * (layers.Count - 1);
        }


        public override void Execute(uint trainingStep)
        {
            if (trainingStep > totalSteps)
                return;

            uint layerIndex = (trainingStep / (uint)LearningDuration);

            if (trainingStep > 0 && trainingStep % LearningDuration == 0)
            {
                MyLog.INFO.WriteLine("Weights between layers " + (layerIndex - 1) + " and " + layerIndex + " intialized by RBM.");
                m_network.RBMBiasMomentum1.Fill(0);
                m_network.RBMBiasMomentum2.Fill(0);
                m_network.RBMWeightMomentum.Fill(0);
            }

            if (trainingStep >= totalSteps)
            {
                MyLog.INFO.WriteLine("RBM initialization finished after " + trainingStep + " steps.");
                return;
            }


            if (LearningBatchSize == 1)
            {

                if (layerIndex > 0)
                    layers[0].RBMForward(m_network.InputLayer);
                

                //  propagate input forward to the layer that should be trained
                for (int i = 1; i < layerIndex; i++)
                {
                    layers[i].RBMForward(layers[i-1]);
                }

                MyAbstractFLayer previous, visible, hidden;

                if (layerIndex <= 0)
                    previous = m_network.InputLayer;
                else
                    previous = layers[(int)layerIndex - 1];
                visible = layers[(int)layerIndex];
                hidden = layers[(int)layerIndex + 1];

                visible.RBMForwardAndStore(previous);
                hidden.RBMForwardAndStore(visible);

                hidden.RBMSamplePositive(visible);

                // sample back and forth between the two layers
                // k-times with CD-k algorithm
                for (int i = 0; i < ContrastiveDivergenceParameter; i++)
			    {
                    visible.RBMBackward(hidden);
                    hidden.RBMForward(visible);
			    }

                // temporary observers, dangerous:
                ((MyNeuronCopyLayer)layers[0]).UpdateObserver();
                ((MyNeuronLayer)layers[1]).UpdateObserver();

                // compute Negative and update weights based on the k-step sampling
                // using positive (obtained at start of CD-k) and negative (obtained after k steps of CD-k)
                hidden.RBMUpdate(visible, LearningRate, LearningMomentum, WeightDecay);

                

                // scale the energy value based on layer size
                // TODO proper energy
                m_network.Energy.SafeCopyToHost();
                m_network.Energy.Host[0] /= (float)((int)hidden.Output.Count);
                m_network.Energy.SafeCopyToDevice();

                
            }
            else
            {
                throw new NotImplementedException("Batch sizes other than 1 are not supported by RBM yet.");
            }


        }
    }
}
