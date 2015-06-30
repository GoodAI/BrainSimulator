using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using BrainSimulator.Task;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Threading.Tasks;
using YAXLib;
using ManagedCuda;
using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.NeuralNetwork.Tasks;

namespace BrainSimulator.NeuralNetwork.Layers
{
    /// <author>Philip Hilm</author>
    /// <status>Working</status>
    /// <summary>Hidden layer node.</summary>
    /// <description></description>
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
                    PreviousWeightDelta.Count = Neurons * Input.Count; // momentum method
                    PreviousBiasDelta.Count = Neurons; // momentum method

                    // RMSProp allocations
                    MeanSquareWeight.Count = Weights.Count;
                    MeanSquareBias.Count = Bias.Count;

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
        public virtual void CreateTasks()
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
    }
}