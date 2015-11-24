using System;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.BasicNodes.PatternDetect
{
    /// <author>GoodAI</author>
    /// <meta>kk</meta>
    /// <status>Working</status>
    /// <summary>Recurrent network trained by Real-Time Recurrent Learning</summary>
    /// </description>
    [YAXSerializeAs("RNNPatternDetector")]
    class PatternDetectorRNNLayer : MyWorkingNode
    {
        #region I/O MEMORY BLOCKS
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }
        #endregion

        #region PROPERTIES
        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("Structure")]
        public ActivationFunctionType ACTIVATION_FUNCTION { get; set; }

        [MyBrowsable, Category("Structure")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
        public int NetworkCount { get; protected set; }
        #endregion

        #region DECLARATIONS
        [MyPersistable]
        public MyMemoryBlock<float> InputWeights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> InputWeightDeltas { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> RecurrentWeights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> RecurrentWeightDeltas { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> OutputWeights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> OutputWeightDeltas { get; protected set; }

        public MyMemoryBlock<float> InputWeightRTRLDerivatives { get; protected set; }
        public MyMemoryBlock<float> PreviousInputWeightRTRLDerivatives { get; protected set; }

        public MyMemoryBlock<float> RecurrentWeightRTRLDerivatives { get; protected set; }
        public MyMemoryBlock<float> PreviousRecurrentWeightRTRLDerivatives { get; protected set; }

        public MyMemoryBlock<float> HiddenActivations { get; protected set; }
        public MyMemoryBlock<float> OutputActivations { get; protected set; }

        public MyMemoryBlock<float> PreviousHiddenActivations { get; protected set; }

        public MyMemoryBlock<float> HiddenActivationDerivatives { get; protected set; }
        public MyMemoryBlock<float> OutputActivationDerivatives { get; protected set; }

        public MyMemoryBlock<float> OutputDeltas { get; protected set; }
        #endregion

        #region TASKS
        public MyInitNetworkTask InitNetwork { get; protected set; }
        public MyFeedforwardTask Feedforward { get; protected set; }
        public MyRTRLTask RTRL { get; protected set; }
        #endregion

        #region SIGNALS
        public MyResetSignal ResetSignal { get; private set; }
        public class MyResetSignal : MySignal { }
        #endregion

        /// <summary>Initializes network with random weights.</summary>
        [Description("Init Network"), MyTaskInfo(OneShot = true)]
        public class MyInitNetworkTask : MyTask<PatternDetectorRNNLayer>
        {
            private MyCudaKernel m_kernel;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");
                m_kernel.SetupExecution(1);
            }

            public override void Execute()
            {
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.InputWeights.GetDevice(Owner), 0, 0.25f);
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.RecurrentWeights.GetDevice(Owner), 0, 0.25f);
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.OutputWeights.GetDevice(Owner), 0, 0.25f);

                Owner.InputWeightDeltas.Fill(0);
                Owner.RecurrentWeightDeltas.Fill(0);
                Owner.OutputWeightDeltas.Fill(0);

                Owner.HiddenActivations.Fill(0);
                Owner.OutputActivations.Fill(0);
                Owner.PreviousHiddenActivations.Fill(0);

                Owner.HiddenActivationDerivatives.Fill(0);
                Owner.OutputActivationDerivatives.Fill(0);

                Owner.InputWeightRTRLDerivatives.Fill(0);
                Owner.RecurrentWeightRTRLDerivatives.Fill(0);

                Owner.PreviousInputWeightRTRLDerivatives.Fill(0);
                Owner.PreviousRecurrentWeightRTRLDerivatives.Fill(0);
            }
        }

        /// <summary>Performs forward pass in te network.</summary>
        [Description("Feedforward"), MyTaskInfo(OneShot = false)]
        public class MyFeedforwardTask : MyTask<PatternDetectorRNNLayer>
        {
            private MyCudaKernel m_feedForwardHiddenKernel;
            private MyCudaKernel m_feedForwardOutputKernel;

            public override void Init(int nGPU)
            {
                m_feedForwardHiddenKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\FeedforwardKernel", "FeedforwardHiddenKernel");
                m_feedForwardHiddenKernel.SetupExecution(Owner.HIDDEN_UNITS);
                m_feedForwardHiddenKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_feedForwardHiddenKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_feedForwardHiddenKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
                m_feedForwardHiddenKernel.SetConstantVariable("D_ACTIVATION_FUNCTION", (int)Owner.ACTIVATION_FUNCTION);

                m_feedForwardOutputKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\FeedforwardKernel", "FeedforwardOutputKernel");
                m_feedForwardOutputKernel.SetupExecution(Owner.OUTPUT_UNITS);
                m_feedForwardOutputKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_feedForwardOutputKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_feedForwardOutputKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
                m_feedForwardOutputKernel.SetConstantVariable("D_ACTIVATION_FUNCTION", (int)Owner.ACTIVATION_FUNCTION);
            }

            public override void Execute()
            {
                if (Owner.ResetSignal.IsIncomingRised())
                {
                    //reset network's internal state
                    Owner.HiddenActivations.Fill(0);
                }

                //move old hidden layer activations to context layer
                Owner.HiddenActivations.CopyToMemoryBlock(Owner.PreviousHiddenActivations, 0, 0, Owner.HiddenActivations.Count);

                //compute activation of hidden layer
                m_feedForwardHiddenKernel.Run(
                     Owner.Input,
                     Owner.HiddenActivations,
                     Owner.PreviousHiddenActivations,
                     Owner.HiddenActivationDerivatives,
                     Owner.InputWeights,
                     Owner.RecurrentWeights
                    );

                //compute activation of output layer
                m_feedForwardOutputKernel.Run(
                    Owner.HiddenActivations,
                    Owner.OutputActivations,
                    Owner.OutputActivationDerivatives,
                    Owner.OutputWeights
                    );

                Owner.OutputActivations.CopyToMemoryBlock(Owner.Output, 0, 0, Owner.OUTPUT_UNITS);
            }
        }

        /// <summary>Computes RTRL partial derivatives and updates network weights.</summary>
        [Description("Real-time recurrent learning"), MyTaskInfo(OneShot = false)]
        public class MyRTRLTask : MyTask<PatternDetectorRNNLayer>
        {
            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 0.5f), YAXElementFor("Structure")]
            public float LEARNING_RATE { get; set; }

            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = 0), YAXElementFor("Structure")]
            public float MOMENTUM_RATE { get; set; }

            private MyCudaKernel m_inputWeightRTRLDerivativesKernel;
            private MyCudaKernel m_recurrentWeightRTRLDerivativesKernel;
            private MyCudaKernel m_outputDeltaKernel;
            private MyCudaKernel m_changeInputWeightsKernel;
            private MyCudaKernel m_changeRecurrentWeightsKernel;
            private MyCudaKernel m_changeOutputWeightsKernel;

            public override void Init(int nGPU)
            {
                m_inputWeightRTRLDerivativesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\RTRLDerivativeKernel", "InputWeightsRTRLDerivativesKernel");
                m_inputWeightRTRLDerivativesKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS * Owner.INPUT_UNITS);
                m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_inputWeightRTRLDerivativesKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

                m_recurrentWeightRTRLDerivativesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\RTRLDerivativeKernel", "RecurrentWeightsRTRLDerivativesKernel");
                m_recurrentWeightRTRLDerivativesKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS);
                m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_recurrentWeightRTRLDerivativesKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

                m_outputDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\OutputDeltaKernel");
                m_outputDeltaKernel.SetupExecution(Owner.OUTPUT_UNITS);
                m_outputDeltaKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_outputDeltaKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_outputDeltaKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

                m_changeInputWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\ChangeWeightsKernel", "ChangeInputWeightsKernel");
                m_changeInputWeightsKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.INPUT_UNITS);
                m_changeInputWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_changeInputWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_changeInputWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

                m_changeRecurrentWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\ChangeWeightsKernel", "ChangeRecurrentWeightsKernel");
                m_changeRecurrentWeightsKernel.SetupExecution(Owner.HIDDEN_UNITS * Owner.HIDDEN_UNITS);
                m_changeRecurrentWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_changeRecurrentWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_changeRecurrentWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);

                m_changeOutputWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Motor\ChangeWeightsKernel", "ChangeOutputWeightsKernel");
                m_changeOutputWeightsKernel.SetupExecution(Owner.OUTPUT_UNITS * Owner.HIDDEN_UNITS);
                m_changeOutputWeightsKernel.SetConstantVariable("D_INPUT_UNITS", Owner.INPUT_UNITS);
                m_changeOutputWeightsKernel.SetConstantVariable("D_HIDDEN_UNITS", Owner.HIDDEN_UNITS);
                m_changeOutputWeightsKernel.SetConstantVariable("D_OUTPUT_UNITS", Owner.OUTPUT_UNITS);
            }

            public override void Execute()
            {
                //move old RTRL partial derivatives
                Owner.InputWeightRTRLDerivatives.CopyToMemoryBlock(Owner.PreviousInputWeightRTRLDerivatives, 0, 0, Owner.InputWeightRTRLDerivatives.Count);
                Owner.RecurrentWeightRTRLDerivatives.CopyToMemoryBlock(Owner.PreviousRecurrentWeightRTRLDerivatives, 0, 0, Owner.RecurrentWeightRTRLDerivatives.Count);

                //compute new RTRL partial derivatives
                m_inputWeightRTRLDerivativesKernel.Run(
                    Owner.Input,
                    Owner.HiddenActivationDerivatives,
                    Owner.RecurrentWeights,
                    Owner.InputWeightRTRLDerivatives,
                    Owner.PreviousInputWeightRTRLDerivatives
                );

                m_recurrentWeightRTRLDerivativesKernel.Run(
                    Owner.PreviousHiddenActivations,
                    Owner.HiddenActivationDerivatives,
                    Owner.RecurrentWeights,
                    Owner.RecurrentWeightRTRLDerivatives,
                    Owner.PreviousRecurrentWeightRTRLDerivatives
                );

                //get delta of output neurons
                m_outputDeltaKernel.Run(
                    Owner.OutputDeltas,
                    Owner.Target,
                    Owner.OutputActivations,
                    Owner.OutputActivationDerivatives
                    );

                //update weights
                m_changeInputWeightsKernel.Run(
                    Owner.InputWeights,
                    Owner.InputWeightDeltas,
                    Owner.OutputWeights,
                    Owner.OutputDeltas,
                    Owner.InputWeightRTRLDerivatives,
                    LEARNING_RATE,
                    MOMENTUM_RATE
                    );

                m_changeRecurrentWeightsKernel.Run(
                    Owner.RecurrentWeights,
                    Owner.RecurrentWeightDeltas,
                    Owner.OutputWeights,
                    Owner.OutputDeltas,
                    Owner.RecurrentWeightRTRLDerivatives,
                    LEARNING_RATE,
                    MOMENTUM_RATE
                    );

                m_changeOutputWeightsKernel.Run(
                    Owner.OutputWeights,
                    Owner.OutputWeightDeltas,
                    Owner.OutputDeltas,
                    Owner.HiddenActivations,
                    LEARNING_RATE,
                    MOMENTUM_RATE
                    );
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Input != null && Target != null)
            {
                INPUT_UNITS = Input.Count;
                OUTPUT_UNITS = Target.Count;
                Output.Count = OUTPUT_UNITS;

                HiddenActivations.Count = HIDDEN_UNITS;
                HiddenActivationDerivatives.Count = HIDDEN_UNITS;
                PreviousHiddenActivations.Count = HIDDEN_UNITS;
                OutputActivations.Count = OUTPUT_UNITS;
                OutputActivationDerivatives.Count = OUTPUT_UNITS;
                OutputDeltas.Count = OUTPUT_UNITS;

                InputWeights.Count = HIDDEN_UNITS * INPUT_UNITS;
                InputWeightDeltas.Count = HIDDEN_UNITS * INPUT_UNITS;

                RecurrentWeights.Count = HIDDEN_UNITS * HIDDEN_UNITS;
                RecurrentWeightDeltas.Count = HIDDEN_UNITS * HIDDEN_UNITS;

                OutputWeights.Count = OUTPUT_UNITS * HIDDEN_UNITS;
                OutputWeightDeltas.Count = OUTPUT_UNITS * HIDDEN_UNITS;

                InputWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * INPUT_UNITS;
                PreviousInputWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * INPUT_UNITS;

                RecurrentWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * HIDDEN_UNITS;
                PreviousRecurrentWeightRTRLDerivatives.Count = HIDDEN_UNITS * HIDDEN_UNITS * HIDDEN_UNITS;


                // make an even number of weights for the cuda random initialisation
                if (InputWeights.Count % 2 != 0)
                    InputWeights.Count++;
                if (RecurrentWeights.Count % 2 != 0)
                    RecurrentWeights.Count++;
                if (OutputWeights.Count % 2 != 0)
                    OutputWeights.Count++;
            }
        }
    }
}
