using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Task;
using XmlFeedForwardNet.Layers;
using XmlFeedForwardNet.Tasks;
using  XmlFeedForwardNet.Tasks.BackPropAgent;
using GoodAI.Core.Utils;
using  XmlFeedForwardNet.Networks;

namespace  XmlFeedForwardNet.Tasks.BackPropAgent
{
    public abstract class MyBackPropAgent
    {
        public int LearningBatchSize;
        public uint LearningDuration;
        public float LearningRate;
        public float LearningMomentum;

        protected MyAbstractFeedForwardNode m_network;

        public MyBackPropAgent(MyAbstractFeedForwardNode network)
        {
            m_network = network;
        }


        public abstract void Execute(uint trainingStep);
    }
}
