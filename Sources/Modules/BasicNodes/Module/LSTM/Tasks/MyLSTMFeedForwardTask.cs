using GoodAI.Core;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System.ComponentModel;
using YAXLib;


namespace GoodAI.Modules.LSTM.Tasks
{
    /// <summary>Performs forward pass in the layer. <br />
    /// Parameters:
    /// <ul>
    ///     <li>CLIP_CELL_STATE: Limits cell states into [-CLIP_CELL_STATE,CLIP_CELL_STATE] interval. Set to 0 for no bounds</li>
    /// </ul>
    /// </summary>
    [Description("Feed forward"), MyTaskInfo(OneShot = false)]
    public class MyLSTMFeedForwardTask : MyAbstractForwardTask<MyLSTMLayer>
    {
        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tLayer")]
        public bool INITIALIZE_SEQUENCE { get; set; }

        [YAXSerializableField(DefaultValue = 0f)]
        [MyBrowsable, Category("\tLayer")]
        public float CLIP_CELL_STATE { get; set; }

        private MyCudaKernel m_feedForwardKernel;

        public MyLSTMFeedForwardTask() { }

        public override void Init(int nGPU)
        {
            switch (Owner.LearningTasks)
            {
                case MyLSTMLayer.LearningTasksType.RTRL:
                    m_feedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "LSTMFeedForwardKernel");
                    break;
                case MyLSTMLayer.LearningTasksType.BPTT:
                    m_feedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "LSTMFeedForwardKernelBPTT");
                    break;
            }
            m_feedForwardKernel.SetupExecution(Owner.MemoryBlocks);
        }

        public override void Execute()
        {
            switch (Owner.LearningTasks)
            {
                case MyLSTMLayer.LearningTasksType.RTRL:
                {
                    if (Owner.ResetSignal.IsIncomingRised())
                    {
                        Owner.Output.Fill(0);
                        Owner.PreviousOutput.Fill(0);
                        Owner.CellStates.Fill(0);
                        Owner.PreviousCellStates.Fill(0);
                    }

                    m_feedForwardKernel.Run(
                        (int)Owner.InputActivationFunction,
                        (int)Owner.GateActivationFunction,
                        Owner.Input,
                        Owner.Output,
                        Owner.PreviousOutput,
                        Owner.CellStates,
                        Owner.PreviousCellStates,
                        Owner.CellInputActivations,
                        Owner.CellInputActivationDerivatives,
                        Owner.InputGateActivations,
                        Owner.InputGateActivationDerivatives,
                        Owner.ForgetGateActivations,
                        Owner.ForgetGateActivationDerivatives,
                        Owner.OutputGateActivations,
                        Owner.OutputGateActivationDerivatives,

                        Owner.CellInputWeights,
                        Owner.InputGateWeights,
                        Owner.ForgetGateWeights,
                        Owner.OutputGateWeights,

                        CLIP_CELL_STATE,
                        Owner.Input.Count,
                        Owner.CellStates.Count,
                        Owner.CellsPerBlock
                    );

                    Owner.CellStates.CopyToMemoryBlock(Owner.PreviousCellStates, 0, 0, Owner.CellStates.Count);
                    Owner.Output.CopyToMemoryBlock(Owner.PreviousOutput, 0, 0, Owner.Output.Count);
                    break;
                }
                case MyLSTMLayer.LearningTasksType.BPTT:
                {
                    if (Owner.ResetSignal.IsIncomingRised())
                    {
                        Owner.Delta.FillAll(0);
                        Owner.Output.FillAll(0);
                        Owner.CellStates.FillAll(0);

                        Owner.CellInputActivations.FillAll(0);
                        Owner.CellStateActivations.FillAll(0);
                        Owner.InputGateActivations.FillAll(0);
                        Owner.ForgetGateActivations.FillAll(0);
                        Owner.OutputGateActivations.FillAll(0);
                        
                        Owner.CellInputActivationDerivatives.FillAll(0);
                        Owner.CellStateActivationDerivatives.FillAll(0);
                        Owner.InputGateActivationDerivatives.FillAll(0);
                        Owner.ForgetGateActivationDerivatives.FillAll(0);
                        Owner.OutputGateActivationDerivatives.FillAll(0);
                        
                        Owner.CellInputWeightGradient.FillAll(0);
                        Owner.OutputGateWeightGradient.FillAll(0);
                        Owner.InputGateWeightGradient.FillAll(0);
                        Owner.ForgetGateWeightGradient.FillAll(0);
                        
                        Owner.CellStateErrors.FillAll(0);
                        Owner.CellInputDeltas.FillAll(0);
                        Owner.OutputGateDeltas.FillAll(0);
                        Owner.ForgetGateDeltas.FillAll(0);
                        Owner.InputGateDeltas.FillAll(0);
                    }

                    if (Owner.ParentNetwork.TimeStep == 0 && INITIALIZE_SEQUENCE)
                    {
                        m_feedForwardKernel.Run(
                            (int)Owner.InputActivationFunction,
                            (int)Owner.GateActivationFunction,
                            Owner.Input,
                            Owner.Output.GetTimeShiftedBlock(-1),
                            Owner.Output.GetTimeShiftedBlock(-1),
                            Owner.CellStates.GetTimeShiftedBlock(-1),
                            Owner.CellStateActivations.GetTimeShiftedBlock(-1),
                            Owner.CellStateActivationDerivatives.GetTimeShiftedBlock(-1),
                            Owner.CellStates.GetTimeShiftedBlock(-1),
                            Owner.CellInputActivations.GetTimeShiftedBlock(-1),
                            Owner.CellInputActivationDerivatives.GetTimeShiftedBlock(-1),
                            Owner.InputGateActivations.GetTimeShiftedBlock(-1),
                            Owner.InputGateActivationDerivatives.GetTimeShiftedBlock(-1),
                            Owner.ForgetGateActivations.GetTimeShiftedBlock(-1),
                            Owner.ForgetGateActivationDerivatives.GetTimeShiftedBlock(-1),
                            Owner.OutputGateActivations.GetTimeShiftedBlock(-1),
                            Owner.OutputGateActivationDerivatives.GetTimeShiftedBlock(-1),

                            Owner.CellInputWeights,
                            Owner.InputGateWeights,
                            Owner.ForgetGateWeights,
                            Owner.OutputGateWeights,

                            CLIP_CELL_STATE,
                            Owner.Input.Count,
                            Owner.CellStates.Count,
                            Owner.CellsPerBlock
                        );
                    }

                    m_feedForwardKernel.Run(
                        (int) Owner.InputActivationFunction,
                        (int) Owner.GateActivationFunction,
                        Owner.Input,
                        Owner.Output,
                        Owner.Output.GetTimeShiftedBlock(-1),
                        Owner.CellStates,
                        Owner.CellStateActivations,
                        Owner.CellStateActivationDerivatives,
                        Owner.CellStates.GetTimeShiftedBlock(-1),
                        Owner.CellInputActivations,
                        Owner.CellInputActivationDerivatives,
                        Owner.InputGateActivations,
                        Owner.InputGateActivationDerivatives,
                        Owner.ForgetGateActivations,
                        Owner.ForgetGateActivationDerivatives,
                        Owner.OutputGateActivations,
                        Owner.OutputGateActivationDerivatives,

                        Owner.CellInputWeights,
                        Owner.InputGateWeights,
                        Owner.ForgetGateWeights,
                        Owner.OutputGateWeights,
                
                        CLIP_CELL_STATE,
                        Owner.Input.Count,
                        Owner.CellStates.Count,
                        Owner.CellsPerBlock
                    );
                    break;
                }
            }
        }
    }
}

