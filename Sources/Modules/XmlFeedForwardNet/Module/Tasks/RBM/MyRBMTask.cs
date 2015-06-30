using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaRand;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using System.ComponentModel;
using YAXLib;
using  XmlFeedForwardNet.Networks;
using  XmlFeedForwardNet.Tasks.BackPropAgent;

namespace  XmlFeedForwardNet.Tasks.RBM
{
    /// <summary>
    /// This task performs the RBM unsupervised initialization of the weights.
    /// </summary>
    [Description("RBM initialization"), MyTaskInfo(Order = 100)]
    public class MyRBMTask : MyTask<MyFeedForwardNode>
    {
        public Int32 NGPU { get; private set; }


        [YAXSerializableField(DefaultValue = MyRBMActivationMode.BINARY)]
        [MyBrowsable, Category("\tLearning"), Description("Set the RBM output mode.\nBINARY: Always set output to 0 or 1 based on the activation probability.\nPROBABILISTIC: The probability itself will be the output.")]
        public MyRBMActivationMode OutputMode { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        private int m_learningBatchSize = 1;
        [MyBrowsable, Category("\tLearning"), Description("Size of the learning batch.\nCurrently, only 1 is supported.")]
        public int LearningBatchSize
        {
            get { return m_learningBatchSize; }
            set
            {
                if (value < 1)
                    return;
                m_learningBatchSize = value;
                if (RBMAgent != null)
                    RBMAgent.LearningBatchSize = value;
            }
        }


        [YAXSerializableField(DefaultValue = (uint)1)]
        private uint m_contrastiveDivergenceParameter = 1;
        [MyBrowsable, Category("\tLearning"), DisplayName("k in CD-k algorithm"), Description("Value of k correponds to CD-k algorithm.")]
        public uint ContrastiveDivergenceParameter
        {
            get { return m_contrastiveDivergenceParameter; }
            set
            {
                if (value < 1)
                    return;
                m_contrastiveDivergenceParameter = value;
                if (RBMAgent != null)
                    RBMAgent.ContrastiveDivergenceParameter = value;
            }
        }

        [YAXSerializableField(DefaultValue = (uint)128)]
        private uint m_learningDuration = 128;
        [MyBrowsable, Category("\tLearning"), DisplayName("Iterations per layer pair"), Description("Stop the RBM training of each pair of layers after this amount of steps.")]
        public uint LearningDuration
        {
            get { return m_learningDuration; }
            set
            {
                if (value < 1)
                    return;
                m_learningDuration = value;
                if (RBMAgent != null)
                    RBMAgent.LearningRate = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0.5f)]
        private float m_learningRate = 0.5f;
        [MyBrowsable, Category("\tLearning"), Description("Factor of weight changes.")]
        public float LearningRate
        {
            get { return m_learningRate; }
            set
            {
                if (value < 0)
                    return;
                m_learningRate = value;
                if (RBMAgent != null)
                    RBMAgent.LearningRate = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0.9f)]
        private float m_momentum = 0.9f;
        [MyBrowsable, Category("\tLearning"), Description("Momentum parameter.")]
        public float Momentum
        {
            get { return m_momentum; }
            set
            {
                if (value < 0)
                    return;
                m_momentum = value;
                if (RBMAgent != null)
                    RBMAgent.LearningMomentum = value;
            }
        }

        [YAXSerializableField(DefaultValue = 0.0001f)]
        private float m_weightDecay = 0.0001f;
        [MyBrowsable, Category("\tLearning"), Description("Weight decay.")]
        public float WeightDecay
        {
            get { return m_weightDecay; }
            set
            {
                if (value < 0)
                    return;
                m_weightDecay = value;
                if (RBMAgent != null)
                    RBMAgent.WeightDecay = value;
            }
        }

        MyRBMAgent RBMAgent;

        private uint m_trainingStep;

        public override void Init(Int32 nGPU)
        {
            m_trainingStep = 0;

            if (Owner.LearningMethod == MyAbstractFeedForwardNode.MyLearningMethod.GRADIENT_DESCENT)
            {
                MyRBMAgent agent = new MyRBMAgent(Owner, nGPU, Owner.LabelInput, LearningDuration);
                agent.LearningBatchSize = LearningBatchSize;
                agent.LearningDuration = LearningDuration;
                agent.LearningRate = LearningRate;
                agent.ContrastiveDivergenceParameter = ContrastiveDivergenceParameter;

                RBMAgent = agent;
            }
            else
            {
                throw new NotImplementedException("Unknown learning method.");
            }
        }
        public override void Execute()
        {
            Owner.InputLayer.Forward();

            RBMAgent.Execute(m_trainingStep);
            m_trainingStep++;

        }
    }
}