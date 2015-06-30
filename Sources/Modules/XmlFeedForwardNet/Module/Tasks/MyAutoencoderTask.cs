using  XmlFeedForwardNet.Networks;
using  XmlFeedForwardNet.Tasks.BackPropAgent;
using BrainSimulator.Memory;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace  XmlFeedForwardNet.Tasks
{
    public class MyAutoencoderTask : MyTask<MyAutoencoderNode>
    {
        [YAXSerializableField(DefaultValue = MyAutoencoderMode.TRAINING)]
        private MyAutoencoderMode m_networkMode;
        [MyBrowsable, Category("\tLearning"), Description("Set the mode of the feedforward network.\nTRAIN: Train the network from input images.\nFORWARD_PASS: Make one pass through the network producing output. Doesn't learn.\nFEATURE_ENCODING: Encode the input into low-dimensional features.\nFEATURE_DECODING: Reconstruct the image from features.")]
        public MyAutoencoderMode NetworkMode
        {
            get { return m_networkMode; }
            set
            {
                if (value != MyAutoencoderMode.FEATURE_DECODING || Owner.FeatureInput != null)
                {
                    m_networkMode = value;
                }
            }
        }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_learningBatchSize = 1;
        [MyBrowsable, Category("\tLearning"), Description("Size of the learning batch.\nIf the batch is 1, then it's an online learning")]
        public int LearningBatchSize
        {
            get { return m_learningBatchSize; }
            set
            {
                if (value > 0)
                    m_learningBatchSize = value;
                if (BackPropAgent != null)
                    BackPropAgent.LearningBatchSize = value;
            }
        }

        [YAXSerializableField(DefaultValue = (uint)0), YAXSerializeAs("LearningDuration")]
        private uint m_learningDuration = 0;
        [MyBrowsable, Category("\tLearning"), Description("Stop the backpropagation after this amount of steps")]
        public uint LearningDuration
        {
            get { return m_learningDuration; }
            set
            {
                m_learningDuration = value;
                if (BackPropAgent != null)
                    BackPropAgent.LearningDuration = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0.001f)]
        private float m_learningRate = 0.001f;
        [MyBrowsable, Category("\tLearning"), Description("Factor of weight changes")]
        public float LearningRate
        {
            get { return m_learningRate; }
            set
            {
                if (value >= 0)
                    m_learningRate = value;
                if (BackPropAgent != null)
                    BackPropAgent.LearningRate = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0)]
        private float m_learningMomentum = 0;
        [MyBrowsable, Category("\tLearning"), Description("Inertia of learning direction")]
        public float LearningMomentum
        {
            get { return m_learningMomentum; }
            set
            {
                if (value < 0)
                    return;
                m_learningMomentum = value;
                if (BackPropAgent != null)
                    BackPropAgent.LearningMomentum = value;
            }
        }

        public MyGradientBackPropAgent BackPropAgent;

        private uint m_trainingStep;

        public override void Init(Int32 nGPU)
        {
            m_trainingStep = 0;

            if (Owner.LearningMethod == MyAbstractFeedForwardNode.MyLearningMethod.GRADIENT_DESCENT)
            {
                BackPropAgent = new MyGradientBackPropAgent(Owner, nGPU, Owner.DataInput); ;
                BackPropAgent.LearningBatchSize = LearningBatchSize;
                BackPropAgent.LearningDuration = LearningDuration;
                BackPropAgent.LearningRate = LearningRate;
                BackPropAgent.LearningMomentum = LearningMomentum;
            }
            else
            {
                throw new NotImplementedException("Unknown learning method.");
            }
        }

        public override void Execute()
        {
            // Reset the input sample
            Owner.ResetSample();

            int startIndex = NetworkMode == MyAutoencoderMode.FEATURE_ENCODING ? (0) : (Owner.FeatureLayerPosition + 1);
            int endIndex = NetworkMode == MyAutoencoderMode.FEATURE_ENCODING ? (Owner.FeatureLayerPosition) : (Owner.FeatureLayerPosition + 1);

            // For every sample...
            for (int i = 0; i < Owner.ForwardSamplesPerStep; i++)
            {
                // Forward propagation
                Owner.ForwardPropagation();

                if (NetworkMode == MyAutoencoderMode.TRAINING)
                {
                    // Backward propagation
                    BackPropAgent.Execute(m_trainingStep);
                }

                // Switch to the next input
                if (i < Owner.ForwardSamplesPerStep - 1)
                {
                    Owner.NextSample();
                }

                m_trainingStep++;
            }
        }
    }
}
