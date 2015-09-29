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
                    Weights.Count = Neurons * Input.Count;
                    Bias.Count = Neurons;

                    // SGD allocations
                    Delta.Count = Neurons;
                    Delta.Mode =  MyTemporalMemoryBlock<float>.ModeType.Cumulate;
                    PreviousWeightDelta.Count = Neurons * Input.Count; // momentum method
                    PreviousBiasDelta.Count = Neurons; // momentum method

                    // RMSProp allocations
                    MeanSquareWeight.Count = Weights.Count;
                    MeanSquareBias.Count = Bias.Count;

                    // Adadelta allocation
                    //AdadeltaWeight.Count = Weights.Count;
                    //AdadeltaBias.Count = Bias.Count;

                    //// vSGD-fd allocations
                    //OriginalWeights.Count = Weights.Count;
                    //OriginalBias.Count = Bias.Count;
                    //OriginalDelta.Count = Delta.Count;
                    //WeightsGrad.Count = Weights.Count;
                    //OriginalWeightsGrad.Count = Weights.Count;
                    //WeightGradCurve.Count = Weights.Count;
                    //AvgWeightGrad.Count = Weights.Count;
                    //AvgWeightGradVar.Count = Weights.Count;
                    //AvgWeightGradCurve.Count = Weights.Count;
                    //AvgWeightGradCurveVar.Count = Weights.Count;
                    //WeightLearningRate.Count = Weights.Count;
                    //WeightMemorySize.Count = Weights.Count;

                    //BiasGrad.Count = Bias.Count;
                    //OriginalBiasGrad.Count = Bias.Count;
                    //BiasGradCurve.Count = Bias.Count;
                    //AvgBiasGrad.Count = Bias.Count;
                    //AvgBiasGradVar.Count = Bias.Count;
                    //AvgBiasGradCurve.Count = Bias.Count;
                    //AvgBiasGradCurveVar.Count = Bias.Count;
                    //BiasLearningRate.Count = Bias.Count;
                    //BiasMemorySize.Count = Bias.Count;
                }
            }
        }


        // Tasks
        public MyFCUpdateWeightsTask UpdateWeights { get; protected set; }
        // PRETRAINING
        //public override void CreateTasks()
        //{
        //    base.CreateTasks();
        //    ForwardTask = new MyFCForwardTask();
        //    DeltaBackTask = new MyFCBackDeltaTask();
        //}
        public void CreateTasks()
        {
            ForwardTask = new MyFCForwardTask();
            DeltaBackTask = new MyFCBackDeltaTask();
        }

        //PRETRAINING
        //public override void DisableLearningTasks()
        //{
        //    base.DisableLearningTasks();

        //    UpdateWeights.Enabled = false;
        //}

        //PRETRAINING
        //public override void EnableLearningTasks()
        //{
        //    base.EnableLearningTasks();

        //    UpdateWeights.Enabled = true;
        //}

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
    }
}