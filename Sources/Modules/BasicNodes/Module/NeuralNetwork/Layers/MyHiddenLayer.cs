using GoodAI.Core.Nodes;
using GoodAI.Modules.NeuralNetwork.Tasks;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Hidden layer node.</summary>
    /// <description>
    /// This is one of the most commonly used layers within Neural Networks./<br></br>
    /// It takes an input and feeds another layer, which can be either an output layer or another hidden layer.<br></br>
    /// The capacity of the network can be scaled by the number of neurons in each layer or by placing multiple layers in succession (deep networks).
    /// </description>
    public class MyHiddenLayer : MyAbstractWeightLayer, IMyCustomTaskFactory
    {
        public override ConnectionType Connection
        {
            get { return ConnectionType.FULLY_CONNECTED; }
        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            if (Neurons > 0)
            {
                if (Input != null)
                {
                    // parameter allocations
                    Weights.Count = Neurons * Input.Count / ParentNetwork.BatchSize;
                    Bias.Count = Neurons;

                    // SGD allocations
                    Delta.Count = Neurons * ParentNetwork.BatchSize;
                    Delta.Mode =  MyTemporalMemoryBlock<float>.ModeType.Cumulate;
                    PreviousWeightDelta.Count = Weights.Count; // momentum method
                    PreviousBiasDelta.Count = Neurons; // momentum method

                    // RMSProp allocations
                    MeanSquareWeight.Count = Weights.Count;
                    MeanSquareBias.Count = Bias.Count;
                }
            }
        }


        // Tasks
        public MyFCUpdateWeightsTask UpdateWeights { get; protected set; }

        public void CreateTasks()
        {
            ForwardTask = new MyFCForwardTask();
            DeltaBackTask = new MyFCBackDeltaTask();
        }

        // Parameterless constructor
        public MyHiddenLayer() { }

        // description
        public override string Description
        {
            get
            {
                return "Hidden layer";
            }
        }

        public override bool SupportsBatchLearning { get { return true; } }
    }
}