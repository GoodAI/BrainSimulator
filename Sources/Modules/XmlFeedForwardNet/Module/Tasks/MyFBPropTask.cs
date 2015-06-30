using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaRand;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using  XmlFeedForwardNet.Tasks.BackPropAgent;
using System.ComponentModel;
using YAXLib;
using  XmlFeedForwardNet.Networks;

namespace  XmlFeedForwardNet.Tasks
{
    /// <summary>
    /// This task performs the forward and backward passes of the feedforward network.
    /// </summary>
    [Description("Forward/Backward Propagation"), MyTaskInfo(Order = 150)]
    public class MyFBPropTask : MyTask<MyFeedForwardNode>
    {
        [YAXSerializableField(DefaultValue = MyFeedForwardMode.TRAINING)]
        [MyBrowsable, Category("\tLearning"), Description("Set the mode of the feedforward network.\nTRAIN: Train the network from input images.\nFORWARD_PASS: Make one pass through the network producing output. Doesn't learn.\nFEATURE_ENCODING: Encode the input into low-dimensional features.\nFEATURE_DECODING: Reconstruct the image from features.")]
        public MyFeedForwardMode NetworkMode { get; set; }

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

        public MyBackPropAgent BackPropAgent;

        private uint m_trainingStep;

        public override void Init(Int32 nGPU)
        {
            m_trainingStep = 0;

            if (Owner.LearningMethod == MyAbstractFeedForwardNode.MyLearningMethod.GRADIENT_DESCENT)
            {
                MyGradientBackPropAgent agent = new MyGradientBackPropAgent(Owner, nGPU, Owner.LabelInput);
                BackPropAgent = agent;
                agent.LearningBatchSize = LearningBatchSize;
                agent.LearningDuration = LearningDuration;
                agent.LearningRate = LearningRate;
                agent.LearningMomentum = LearningMomentum;
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

            // For every sample...
            for (uint i = 0; i < Owner.ForwardSamplesPerStep; i++)
            {
                // Forward propagation
                Owner.ForwardPropagation();

                if (NetworkMode == MyFeedForwardMode.TRAINING)
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
