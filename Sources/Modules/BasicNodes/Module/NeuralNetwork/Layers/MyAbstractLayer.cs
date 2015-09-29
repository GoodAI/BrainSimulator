using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System.ComponentModel;
using YAXLib;
using ManagedCuda.BasicTypes;
using System.Collections.Generic;
using GoodAI.Core.Signals;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    // PRETRAINING
    //public class MyIsLearningSignal : MySignal { }

    public enum ConnectionType
    {
        NOT_SET,
        FULLY_CONNECTED,
        CONVOLUTION,
        ONE_TO_ONE,
        GAUSSIAN,
        PARTIAL_UPDATE
    }

    public enum ActivationFunctionType
    {
        NO_ACTIVATION,
        SIGMOID,
        IDENTITY,
        GAUSSIAN,
        RATIONAL_SIGMOID,
        RELU,
        SOFTMAX,
        TANH,
        LECUN_TANH,
    }

    public abstract class MyAbstractLayer : MyWorkingNode
    {
        // PRETRAINING
        //public MyIsLearningSignal IsLearning { get; set; }

        // Properties
        [YAXSerializableField(DefaultValue = ActivationFunctionType.NO_ACTIVATION)]
        [MyBrowsable, Category("\tLayer")]
        public virtual ActivationFunctionType ActivationFunction { get; set; }

        [YAXSerializableField(DefaultValue = 128)]
        [MyBrowsable, Category("\tLayer"), DisplayName("\tNeurons")]
        public virtual int Neurons { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("Misc")]
        public int OutputColumnHint { get; set; }

        // The preceding layer in the topological ordering
        public MyAbstractLayer PreviousTopologicalLayer { get; set; }
        // The succeeding layer in the topological ordering
        public MyAbstractLayer NextTopologicalLayer { get; set; }

        // layers feeding connections into this layer
        public List<MyAbstractLayer> PreviousConnectedLayers { get; set; }
        // layers in which this layer feeds connections
        public List<MyAbstractLayer> NextConnectedLayers { get; set; }

        public MyNeuralNetworkGroup ParentNetwork
        {
            get { return Parent as MyNeuralNetworkGroup; }
        }


        public virtual ConnectionType Connection
        {
            get { return ConnectionType.NOT_SET; }
        }


        #region Memory blocks
        // Memory blocks
        [MyInputBlock(0)]
        public virtual MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        // PRETRAINING
        //[MyInputBlock(1)]
        //// only host side of the memory is ever used!
        //public virtual MyMemoryBlock<float> CanLearn
        //{
        //    get { return GetInput(1); }
        //}

       [MyOutputBlock(0)]
        public virtual MyTemporalMemoryBlock<float> Output
        {
            get { return GetOutput(0) as MyTemporalMemoryBlock<float>; }
            set { SetOutput(0, value); }
        }

        public virtual MyTemporalMemoryBlock<float> Delta { get; protected set; }
        #endregion

        static public CUdeviceptr DetermineInput(MyAbstractLayer layer)
        {
            if(layer is MyGaussianHiddenLayer)
                return (layer as MyGaussianHiddenLayer).NoisyInput.GetDevicePtr(layer.GPU);
            else if (layer is MyAbstractWeightLayer)
                return (layer as MyAbstractWeightLayer).NeuronInput.GetDevicePtr(layer.GPU);
            else
                return layer.Input.GetDevicePtr(layer.GPU);
        }

        //parameterless constructor
        public MyAbstractLayer()
        {
            PreviousConnectedLayers = new List<MyAbstractLayer>();
            NextConnectedLayers = new List<MyAbstractLayer>();
        }

        // PRETRAINING
        //public virtual void DisableLearningTasks()
        //{
        //    Delta.FillAll(0);
        //    ForwardTask.Enabled = false;
        //    DeltaBackTask.Enabled = false;
        //}

        // PRETRAINING
        //public virtual void EnableLearningTasks()
        //{
        //    ForwardTask.Enabled = true;
        //    DeltaBackTask.Enabled = true;
        //}

        // PRETRAINING
        //public virtual void CreateTasks()
        //{
        //    PretrainingTask = new MyNodePretrainingTask();
        //}

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            Output.Count = Neurons;
            Output.ColumnHint = OutputColumnHint;
            Delta.Count = Neurons;
        }

        public override void Validate(MyValidator validator)
        {
            // base.Validate(validator);
            validator.AssertError(Neurons > 0, this, "Number of neurons should be > 0");
            validator.AssertWarning(Connection != ConnectionType.NOT_SET, this, "ConnectionType not set for " + this);
        }

        // PRETRAINING
        //public MyTask PretrainingTask { get; protected set; }
        public MyTask ForwardTask { get; protected set; }
        public MyTask DeltaBackTask { get; protected set; }
    }
}