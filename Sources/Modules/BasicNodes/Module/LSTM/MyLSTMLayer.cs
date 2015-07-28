using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using YAXLib;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.LSTM.Tasks;

namespace GoodAI.Modules.LSTM
{
    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>Long Short Term Memory layer</summary>
    /// <description>Fully recurrent Long Short Term Memory (LSTM) hidden layer with forget gates and peephole connections trained by truncated Real-Time Recurrent Learning (RTRL) algorithm.<br />
    ///              Parameters:
    ///              <ul>
    ///                 <li>ActivationFunction: Activation function applied to cell input. All gates use sigmoid activation function</li>
    ///                 <li>CellsPerBlock: Number of cells in each LSTM memory block</li>
    ///                 <li>MemoryBlocks: Number of LSTM memory blocks in the layer</li>
    ///                 <li>Neurons: Read-only number of cells in the layer calculated as MemoryBlocks * CellsPerBlock</li>
    ///              </ul>
    ///              Training hyperparameters are set by neural network group and can be configured there.
    /// </description>
    public class MyLSTMLayer : MyAbstractLayer, IMyCustomTaskFactory
    {
        // Properties
        public override int Neurons
        {
            get { return MemoryBlocks * CellsPerBlock; }
            set {}
        }

        public override ConnectionType Connection
        {
            get { return ConnectionType.FULLY_CONNECTED; }
        }

        [YAXSerializableField(DefaultValue = 8)]
        [MyBrowsable, Category("\tLayer")]
        public int MemoryBlocks { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tLayer")]
        public int CellsPerBlock { get; set; }

        //Tasks
        MyLSTMInitLayerTask initLayerTask { get; set; }
        MyLSTMPartialDerivativesTask partialDerivativesTask { get; set; }
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
        public MyMemoryBlock<float> CellInputWeightDeltas { get; set; }
        public MyMemoryBlock<float> CellInputWeightMeanSquares { get; set; } // RMSProp memory
        [MyPersistable]
        public MyMemoryBlock<float> InputGateWeights { get; set; }
        public MyMemoryBlock<float> InputGateWeightDeltas { get; set; }
        public MyMemoryBlock<float> InputGateWeightMeanSquares { get; set; } // RMSProp memory
        [MyPersistable]
        public MyMemoryBlock<float> ForgetGateWeights { get; set; }
        public MyMemoryBlock<float> ForgetGateWeightDeltas { get; set; }
        public MyMemoryBlock<float> ForgetGateWeightMeanSquares { get; set; } // RMSProp memory
        [MyPersistable]
        public MyMemoryBlock<float> OutputGateWeights { get; set; }
        public MyMemoryBlock<float> OutputGateWeightDeltas { get; set; }
        public MyMemoryBlock<float> OutputGateWeightMeanSquares { get; set; } // RMSProp memory

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

            CellInputWeightDeltas.Count = CellInputWeights.Count;
            InputGateWeightDeltas.Count = InputGateWeights.Count;
            ForgetGateWeightDeltas.Count = ForgetGateWeights.Count;
            OutputGateWeightDeltas.Count = OutputGateWeights.Count;

            CellInputWeightMeanSquares.Count = CellInputWeights.Count;
            InputGateWeightMeanSquares.Count = InputGateWeights.Count;
            ForgetGateWeightMeanSquares.Count = ForgetGateWeights.Count;
            OutputGateWeightMeanSquares.Count = OutputGateWeights.Count;

            CellWeightsRTRLPartials.Count = CellInputWeights.Count;
            InputGateWeightsRTRLPartials.Count = InputGateWeights.Count * CellsPerBlock;
            ForgetGateWeightsRTRLPartials.Count = ForgetGateWeights.Count * CellsPerBlock;
            
            CellStateErrors.Count = CellStates.Count;
            OutputGateDeltas.Count = MemoryBlocks;

            Delta.Count = CellStates.Count; // computed by previous layer

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

        public void CreateTasks()
        {
            ForwardTask = new MyLSTMFeedForwardTask();
            DeltaBackTask = new MyLSTMDeltaTask();
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
