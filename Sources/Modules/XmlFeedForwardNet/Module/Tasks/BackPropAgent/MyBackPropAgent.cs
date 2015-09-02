using XmlFeedForwardNet.Networks;

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
