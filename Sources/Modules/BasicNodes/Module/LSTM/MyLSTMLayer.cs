using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using GoodAI.Core.Utils;
using GoodAI.Core.Signals;
using GoodAI.Modules.NeuralNetwork;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;
using YAXLib;
using System;
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
    ///                 <li>InputActivationFunction: Activation function applied to cell input</li>
    ///                 <li>GateActivationFunction: Activation function applied to gate input. Read-only, all gates use sigmoid activation function</li>
    ///                 <li>ActivationFunction: Activation function applied to cell output. Read-only, no activation function is used</li>
    ///                 <li>CellsPerBlock: Number of cells in each LSTM memory block</li>
    ///                 <li>MemoryBlocks: Number of LSTM memory blocks in the layer</li>
    ///                 <li>Neurons: Read-only number of cells in the layer calculated as MemoryBlocks * CellsPerBlock</li>
    ///              </ul>
    ///              
    ///              Signals:
    ///              <ul>
    ///                 <li>Reset: Resets LSTM's internal state to initial value</li>
    ///              </ul>
    /// </description>
    public class MyLSTMLayer : MyAbstractLayer, IMyCustomTaskFactory
    {
        // Properties
        [ReadOnly(true)]
        public override int Neurons
        {
            get { return MemoryBlocks * CellsPerBlock; }
            set {}
        }
        public enum LearningTasksType
        {
            RTRL,
            BPTT
        }

        [YAXSerializableField(DefaultValue = LearningTasksType.RTRL)]
        [MyBrowsable, Category("\tLayer")]
        public LearningTasksType LearningTasks { get; set; }

        [YAXSerializableField(DefaultValue = ActivationFunctionType.TANH)]
        [MyBrowsable, Category("\tLayer")]
        public ActivationFunctionType InputActivationFunction { get; set; }

        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("\tLayer"), ReadOnly(true)]
        public ActivationFunctionType GateActivationFunction { get; set; }

        [MyBrowsable, Category("\tLayer"), ReadOnly(true)]
        public override ActivationFunctionType ActivationFunction
        {
            get { return ActivationFunctionType.NO_ACTIVATION; }
            set { }
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

        public int CellInputSize { get { return Input.Count + Output.Count + 1; } }
        public int GateInputSize { get { return Input.Count + Output.Count + CellsPerBlock + 1; } }

        //Tasks
        protected MyLSTMInitLayerTask initLayerTask { get; set; }
        protected MyLSTMPartialDerivativesTask partialDerivativesTask { get; set; }
        protected MyLSTMUpdateWeightsTask updateWeightsTask { get; set; }

        //Signals
        public MyResetSignal ResetSignal { get; set; }
        public class MyResetSignal : MySignal { }

        // Memory blocks
        public virtual MyMemoryBlock<float> Temporary { get; set; }

        public virtual MyTemporalMemoryBlock<float> CellStates { get; set; }
        public virtual MyMemoryBlock<float> PreviousCellStates { get; set; }

        public virtual MyTemporalMemoryBlock<float> InputGateNetInput { get; set; }
        public virtual MyTemporalMemoryBlock<float> ForgetGateNetInput { get; set; }
        public virtual MyTemporalMemoryBlock<float> OutputGateNetInput { get; set; }
        public virtual MyTemporalMemoryBlock<float> CellInputNetInput { get; set; }

        public virtual MyTemporalMemoryBlock<float> CellInputActivations { get; set; }
        public virtual MyTemporalMemoryBlock<float> CellStateActivations { get; set; }
        public virtual MyTemporalMemoryBlock<float> InputGateActivations { get; set; }
        public virtual MyTemporalMemoryBlock<float> ForgetGateActivations { get; set; }
        public virtual MyTemporalMemoryBlock<float> OutputGateActivations { get; set; }

        public virtual MyTemporalMemoryBlock<float> CellInputActivationDerivatives { get; set; }
        public virtual MyTemporalMemoryBlock<float> CellStateActivationDerivatives { get; set; }
        public virtual MyTemporalMemoryBlock<float> InputGateActivationDerivatives { get; set; }
        public virtual MyTemporalMemoryBlock<float> ForgetGateActivationDerivatives { get; set; }
        public virtual MyTemporalMemoryBlock<float> OutputGateActivationDerivatives { get; set; }

        [MyPersistable]
        public virtual MyMemoryBlock<float> CellInputWeights { get; set; }
        public virtual MyMemoryBlock<float> CellInputWeightDeltas { get; set; }
        public virtual MyMemoryBlock<float> CellInputWeightMeanSquares { get; set; } // RMSProp memory
        [MyPersistable]
        public virtual MyMemoryBlock<float> InputGateWeights { get; set; }
        public virtual MyMemoryBlock<float> InputGateWeightDeltas { get; set; }
        public virtual MyMemoryBlock<float> InputGateWeightMeanSquares { get; set; } // RMSProp memory
        [MyPersistable]
        public virtual MyMemoryBlock<float> ForgetGateWeights { get; set; }
        public virtual MyMemoryBlock<float> ForgetGateWeightDeltas { get; set; }
        public virtual MyMemoryBlock<float> ForgetGateWeightMeanSquares { get; set; } // RMSProp memory
        [MyPersistable]
        public virtual MyMemoryBlock<float> OutputGateWeights { get; set; }
        public virtual MyMemoryBlock<float> OutputGateWeightDeltas { get; set; }
        public virtual MyMemoryBlock<float> OutputGateWeightMeanSquares { get; set; } // RMSProp memory

        public virtual MyTemporalMemoryBlock<float> CellInputWeightGradient { get; set; }
        public virtual MyTemporalMemoryBlock<float> OutputGateWeightGradient { get; set; }
        public virtual MyTemporalMemoryBlock<float> InputGateWeightGradient { get; set; }
        public virtual MyTemporalMemoryBlock<float> ForgetGateWeightGradient { get; set; }

        public virtual MyMemoryBlock<float> CellWeightsRTRLPartials { get; set; }
        public virtual MyMemoryBlock<float> InputGateWeightsRTRLPartials { get; set; }
        public virtual MyMemoryBlock<float> ForgetGateWeightsRTRLPartials { get; set; }

        public virtual MyTemporalMemoryBlock<float> CellStateErrors { get; set; }
        public virtual MyTemporalMemoryBlock<float> CellInputDeltas { get; set; }
        public virtual MyTemporalMemoryBlock<float> OutputGateDeltas { get; set; }
        public virtual MyTemporalMemoryBlock<float> ForgetGateDeltas { get; set; }
        public virtual MyTemporalMemoryBlock<float> InputGateDeltas { get; set; }

        public virtual MyMemoryBlock<float> PreviousOutput { get; set; }

        public override void UpdateMemoryBlocks()
        {
            //---- set paramters for BPTT/RTRL
            if (ParentNetwork != null && ParentNetwork.GroupOutputNodes.Length > 0)
            {
                switch (LearningTasks)
                {
                    case MyLSTMLayer.LearningTasksType.RTRL:
                        ParentNetwork.SequenceLength = 1;
                        // TODO this actually does not allow 0.25 for RTRL, should be improved
                        if (initLayerTask.INIT_WEIGHTS_STDDEV == 0.2f)
                            initLayerTask.INIT_WEIGHTS_STDDEV = 0.025f;
                        //System.Console.WriteLine("LSTM: Udpated Group parameters to RTRL: SequenceLength");
                        break;
                    case MyLSTMLayer.LearningTasksType.BPTT:
                        // TODO this actually does not allow 0.025 for BPTT, should be improved
                        if (initLayerTask.INIT_WEIGHTS_STDDEV == 0.025f)
                            initLayerTask.INIT_WEIGHTS_STDDEV = 0.2f;
                        //ParentNetwork.SequenceLength = 2;
                        //System.Console.WriteLine("LSTM: Udpated Group parameters to BPTT: SequenceLength");
                        break;
                    default:
                        break;
                }
            }

            base.UpdateMemoryBlocks();

            if (Input == null)
                return;

            Temporary.Count = MemoryBlocks * GateInputSize;

            CellStates.Count = MemoryBlocks * CellsPerBlock;
            PreviousCellStates.Count = CellStates.Count;

            Output.Count = CellStates.Count;
            PreviousOutput.Count = CellStates.Count;

            CellInputNetInput.Count = MemoryBlocks * CellsPerBlock;
            InputGateNetInput.Count = MemoryBlocks;
            ForgetGateNetInput.Count = MemoryBlocks;
            OutputGateNetInput.Count = MemoryBlocks;

            CellInputActivations.Count = CellStates.Count;
            CellStateActivations.Count = CellStates.Count;
            InputGateActivations.Count = MemoryBlocks;
            ForgetGateActivations.Count = MemoryBlocks;
            OutputGateActivations.Count = MemoryBlocks;

            CellInputActivationDerivatives.Count = CellStates.Count;
            CellStateActivationDerivatives.Count = CellStates.Count;
            InputGateActivationDerivatives.Count = MemoryBlocks;
            ForgetGateActivationDerivatives.Count = MemoryBlocks;
            OutputGateActivationDerivatives.Count = MemoryBlocks;

            CellInputWeights.Count = CellInputSize * CellStates.Count;
            InputGateWeights.Count = GateInputSize * InputGateActivations.Count;
            ForgetGateWeights.Count = GateInputSize * ForgetGateActivations.Count;
            OutputGateWeights.Count = GateInputSize * OutputGateActivations.Count;

            CellInputWeightGradient.Count = CellInputWeights.Count;
            OutputGateWeightGradient.Count = OutputGateWeights.Count;
            InputGateWeightGradient.Count = InputGateWeights.Count;
            ForgetGateWeightGradient.Count = ForgetGateWeights.Count;

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

            CellInputWeightGradient.Mode = MyTemporalMemoryBlock<float>.ModeType.Cumulate;
            OutputGateWeightGradient.Mode = MyTemporalMemoryBlock<float>.ModeType.Cumulate;
            InputGateWeightGradient.Mode = MyTemporalMemoryBlock<float>.ModeType.Cumulate;
            ForgetGateWeightGradient.Mode = MyTemporalMemoryBlock<float>.ModeType.Cumulate;
            
            CellStateErrors.Count = CellStates.Count;
            CellInputDeltas.Count = CellStates.Count;
            OutputGateDeltas.Count = MemoryBlocks;
            ForgetGateDeltas.Count = MemoryBlocks;
            InputGateDeltas.Count = MemoryBlocks;

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

        public override void CreateTasks()
        {
            base.CreateTasks();

            ForwardTask = new MyLSTMFeedForwardTask();
            DeltaBackTask = new MyLSTMDeltaTask();
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            if (ParentNetwork == null)
            {
                validator.AddError(this, "LSTM layer needs to be in a neural network group");
            }
        }

        public override string Description
        {
            get
            {
                string str = "LSTM Layer";
                switch (LearningTasks)
                {
                    case LearningTasksType.RTRL:
                        str += " (RTRL)";
                        break;
                    case LearningTasksType.BPTT:
                        str += " (BPTT)";
                        break;
                }
                return str;
            }
        }

        public void PrintMemBlock2Console(MyMemoryBlock<float> m, string s = "")
        {
            System.Console.Write("  + " + s + ": ");
            m.SafeCopyToHost();
            for (int i = 0; i < Math.Min(30,m.Count); i++)
			{
                System.Console.Write(m.Host[i]+" ");
            }
            System.Console.WriteLine("");
        }

    }
}
