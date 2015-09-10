using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    public interface IMyForwardTask { }
    public abstract class MyAbstractForwardTask<OwnerType> : MyTask<OwnerType>, IMyForwardTask where OwnerType : MyAbstractLayer
    {

    }

    public interface IMyDeltaTask { }
    public abstract class MyAbstractBackDeltaTask<OwnerType> : MyTask<OwnerType>, IMyDeltaTask where OwnerType : MyAbstractLayer
    {

    }

    public interface IMyOutputDeltaTask { }
    public abstract class MyAbstractLossTask<OwnerType> : MyTask<OwnerType>, IMyOutputDeltaTask where OwnerType : MyAbstractLayer
    {

    }

    public interface IMyUpdateWeightsTask { }
    public abstract class MyAbstractUpdateWeightsTask<OwnerType> : MyTask<OwnerType>, IMyUpdateWeightsTask where OwnerType : MyAbstractLayer
    {

    }

    public abstract class MyBackwardTask<OwnerType> : MyTask<OwnerType> where OwnerType : MyAbstractLayer
    {
    }

    public abstract class MyBackwardWeightTask : MyBackwardTask<MyAbstractLayer>
    {

        // this property is/will be common for all children!
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
            }
        }
    }


    public class MyBackwardNeuralTask : MyBackwardWeightTask
    {
        public override void Init(int nGPU)
        {
            MyLog.DEBUG.WriteLine("Neural Task Init");
        }

        public override void Execute()
        {
            MyLog.DEBUG.WriteLine("Neural Task Execute");
        }
    }

    public class MyBackwardActivationTask : MyBackwardTask<MyAbstractLayer>
    {
        public override void Init(int nGPU)
        {
            MyLog.DEBUG.WriteLine("Activation Task Init");
        }

        public override void Execute()
        {
            MyLog.DEBUG.WriteLine("Activation Task Execute");
        }

        // example usage: define
        // public MyBackwardActivationTask BackwardTask { get; protected set; }
        // in convolution layer node

    }

}
