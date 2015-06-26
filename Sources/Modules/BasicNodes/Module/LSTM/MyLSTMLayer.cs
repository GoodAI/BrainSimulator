using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using YAXLib;
using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.NeuralNetwork.Layers;
using BrainSimulator.LSTM.Tasks;

namespace BrainSimulator.LSTM
{
    /// <author>Karol Kuna</author>
    /// <status>WIP</status>
    /// <summary>Long Short Term Memory layer</summary>
    /// <description></description>
    public class MyLSTMLayer : MyAbstractLayer
    {
        // Properties
        public override int Neurons
        {
            get { return MemoryBlocks * CellsPerBlock; }
            set {}
        }

        [YAXSerializableField(DefaultValue = 8)]
        [MyBrowsable, Category("\tLayer")]
        public int MemoryBlocks { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tLayer")]
        public int CellsPerBlock { get; set; }

        [YAXSerializableField(DefaultValue = 0.1f)]
        [MyBrowsable, Category("\tLayer")]
        public float LearningRate { get; set; }

        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("\tLayer")]
        public ActivationFunctionType ActivationFunction { get; set; }

        //Tasks
        MyLSTMInitLayerTask initLayerTask { get; set; }
        MyLSTMFeedForwardTask feedForwardTask { get; set; }
        MyLSTMPartialDerivativesTask partialDerivativesTask { get; set; }
        MyLSTMDeltaTask deltaTask { get; set; }
        MyLSTMUpdateWeightsTask updateWeightsTask { get; set; }

        // Memory blocks
        public MyMemoryBlock<float> CellStates { get; set; }
        public MyMemoryBlock<float> PreviousCellStates { get; set; }

        public MyMemoryBlock<float> CellInputActivations { get; set; }
        public MyMemoryBlock<float> InputGateActivations { get; set; }
        public MyMemoryBlock<float> ForgetGateActivations { get; set; }
        public MyMemoryBlock<float> OutputGateActivations { get; set; }

        public MyMemoryBlock<float> CellInputActivationDerivatives { get; set; }
        public MyMemoryBlock<float> InputGateActivationDerivatives { get; set; }
        public MyMemoryBlock<float> ForgetGateActivationDerivatives { get; set; }
        public MyMemoryBlock<float> OutputGateActivationDerivatives { get; set; }

        [MyPersistable]
        public MyMemoryBlock<float> CellInputWeights { get; set; }
        [MyPersistable]
        public MyMemoryBlock<float> InputGateWeights { get; set; }
        [MyPersistable]
        public MyMemoryBlock<float> ForgetGateWeights { get; set; }
        [MyPersistable]
        public MyMemoryBlock<float> OutputGateWeights { get; set; }

        public MyMemoryBlock<float> CellWeightsRTRLPartials { get; set; }
        public MyMemoryBlock<float> InputGateWeightsRTRLPartials { get; set; }
        public MyMemoryBlock<float> ForgetGateWeightsRTRLPartials { get; set; }

        public MyMemoryBlock<float> CellStateErrors { get; set; }
        public MyMemoryBlock<float> OutputGateDeltas { get; set; }

        public MyMemoryBlock<float> PreviousOutput { get; set; }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            if (Input == null)
                return;

            CellStates.Count = MemoryBlocks * CellsPerBlock;
            PreviousCellStates.Count = CellStates.Count;

            Output.Count = CellStates.Count;
            PreviousOutput.Count = CellStates.Count;

            CellInputActivations.Count = CellStates.Count;
            InputGateActivations.Count = MemoryBlocks;
            ForgetGateActivations.Count = MemoryBlocks;
            OutputGateActivations.Count = MemoryBlocks;

            CellInputActivationDerivatives.Count = CellStates.Count;
            InputGateActivationDerivatives.Count = MemoryBlocks;
            ForgetGateActivationDerivatives.Count = MemoryBlocks;
            OutputGateActivationDerivatives.Count = MemoryBlocks;

            int cellInputSize = Input.Count + Output.Count + 1;
            int gateInputSize = Input.Count + Output.Count + CellsPerBlock + 1;

            CellInputWeights.Count = cellInputSize * CellStates.Count;
            InputGateWeights.Count = gateInputSize * InputGateActivations.Count;
            ForgetGateWeights.Count = gateInputSize * ForgetGateActivations.Count;
            OutputGateWeights.Count = gateInputSize * OutputGateActivations.Count;

            CellWeightsRTRLPartials.Count = CellInputWeights.Count;
            InputGateWeightsRTRLPartials.Count = InputGateWeights.Count * CellsPerBlock;
            ForgetGateWeightsRTRLPartials.Count = ForgetGateWeights.Count * CellsPerBlock;
            
            CellStateErrors.Count = CellStates.Count;
            OutputGateDeltas.Count = MemoryBlocks;

            // make an even number of weights for the cuda random initialisation
            if (CellInputWeights.Count % 2 != 0)
                CellInputWeights.Count++;
            if (InputGateWeights.Count % 2 != 0)
                InputGateWeights.Count++;
            if (ForgetGateWeights.Count % 2 != 0)
                ForgetGateWeights.Count++;
            if (OutputGateWeights.Count % 2 != 0)
                OutputGateWeights.Count++;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
        }

        public override string Description
        {
            get
            {
                return "LSTM Layer";
            }
        }
    }
}
