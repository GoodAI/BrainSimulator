using BrainSimulator.Memory;
using BrainSimulator.Utils;
using BrainSimulator.Signals;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;
using BrainSimulator.Task;
using BrainSimulator.Nodes;
using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.NeuralNetwork.Tasks;

namespace BrainSimulator.NeuralNetwork.Layers
{
    public abstract class MyAbstractWeightLayer : MyAbstractLayer // use this layer to enable automatic gradient checking
    {
        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("\tLayer")]
        public override ActivationFunctionType ActivationFunction { get; set; }

        #region Memory blocks
        // Memory blocks
        [MyPersistable]
        public MyMemoryBlock<float> Weights { get; set; }
        public MyMemoryBlock<float> PreviousWeightDelta { get; protected set; }

        [MyPersistable]
        public MyMemoryBlock<float> Bias { get; protected set; }
        public MyMemoryBlock<float> PreviousBiasDelta { get; protected set; }

        public MyMemoryBlock<float> NeuronInput { get; protected set; }

        public MyMemoryBlock<float> DropoutMask { get; protected set; }

        public MyMemoryBlock<float> L1Term { get; protected set; }
        public MyMemoryBlock<float> L2Term { get; protected set; }

        // RMSProp memory
        public MyMemoryBlock<float> MeanSquareWeight { get; protected set; }
        public MyMemoryBlock<float> MeanSquareBias { get; protected set; }

        // vSGD-fd memory
        public MyMemoryBlock<float> OriginalWeights { get; protected set; }
        public MyMemoryBlock<float> OriginalBias { get; protected set; }
        public MyMemoryBlock<float> OriginalDelta { get; protected set; }
        public MyMemoryBlock<float> WeightsGrad { get; protected set; }
        public MyMemoryBlock<float> OriginalWeightsGrad { get; protected set; }
        public MyMemoryBlock<float> WeightGradCurve { get; protected set; }
        public MyMemoryBlock<float> AvgWeightGrad { get; protected set; }
        public MyMemoryBlock<float> AvgWeightGradVar { get; protected set; }
        public MyMemoryBlock<float> AvgWeightGradCurve { get; protected set; }
        public MyMemoryBlock<float> AvgWeightGradCurveVar { get; protected set; }
        public MyMemoryBlock<float> WeightLearningRate { get; protected set; }
        public MyMemoryBlock<float> WeightMemorySize { get; protected set; }

        public MyMemoryBlock<float> BiasGrad { get; protected set; }
        public MyMemoryBlock<float> OriginalBiasGrad { get; protected set; }
        public MyMemoryBlock<float> BiasGradCurve { get; protected set; }
        public MyMemoryBlock<float> AvgBiasGrad { get; protected set; }
        public MyMemoryBlock<float> AvgBiasGradVar { get; protected set; }
        public MyMemoryBlock<float> AvgBiasGradCurve { get; protected set; }
        public MyMemoryBlock<float> AvgBiasGradCurveVar { get; protected set; }
        public MyMemoryBlock<float> BiasLearningRate { get; protected set; }
        public MyMemoryBlock<float> BiasMemorySize { get; protected set; }

        #endregion

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            NeuronInput.Count = Neurons;
            if (Neurons % 2 == 0)
                DropoutMask.Count = Neurons;
            else
                DropoutMask.Count = Neurons + 1;
            L1Term.Count = 1;
            L2Term.Count = 1;
        }

        // Tasks
        public MyInitWeightsTask InitWeights { get; protected set; }
        public MyCreateDropoutMaskTask CreateDropoutMask { get; protected set; }

        //parameterless constructor
        public MyAbstractWeightLayer() { }
    }
}