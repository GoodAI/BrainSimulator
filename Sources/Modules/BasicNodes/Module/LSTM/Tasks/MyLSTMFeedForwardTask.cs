using GoodAI.Core;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using GoodAI.Modules.NeuralNetwork.Tasks;
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;
using System;
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

        private CUdeviceptr nullptr = new CUdeviceptr(0);

        private MyCudaKernel m_feedForwardKernel;

        private MyCudaKernel m_netInputFeedForwardKernel;

        private MyCudaKernel m_cellStateFeedForwardKernel;
        private MyCudaKernel m_outputStateFeedForwardKernel;

        public MyLSTMFeedForwardTask() { }

        public override void Init(int nGPU)
        {
            switch (Owner.LearningTasks)
            {
                case MyLSTMLayer.LearningTasksType.RTRL:
                    m_feedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "LSTMFeedForwardKernel");
                    m_feedForwardKernel.SetupExecution(Owner.MemoryBlocks);
                    break;
                
                case MyLSTMLayer.LearningTasksType.BPTT:
                    m_netInputFeedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "GetNetInput");
                    m_netInputFeedForwardKernel.DynamicSharedMemory = 512 * sizeof(float);
                    m_netInputFeedForwardKernel.GridDimensions = Owner.MemoryBlocks;
                    m_netInputFeedForwardKernel.BlockDimensions = 512;

                    m_cellStateFeedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "CellStateFeedForwardKernelBPTT");
                    m_cellStateFeedForwardKernel.BlockDimensions = Owner.MemoryBlocks;

                    m_outputStateFeedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "OutputStateFeedForwardKernelBPTT");
                    m_outputStateFeedForwardKernel.BlockDimensions = Owner.MemoryBlocks;
                    break;
            }
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
                        (int)Owner.ActivationFunction,
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
                        m_netInputFeedForwardKernel.Run(
                            Owner.InputGateNetInput.GetTimeShiftedBlock(-1),
                            Owner.Temporary,
                            Owner.CellsPerBlock,
                            Owner.InputGateWeights,
                            Owner.Input,
                            Owner.Input.Count,
                            Owner.Output.GetTimeShiftedBlock(-1),
                            Owner.Output.Count,
                            Owner.CellStates.GetTimeShiftedBlock(-1),
                            1,
                            1
                        );

                        m_netInputFeedForwardKernel.Run(
                            Owner.ForgetGateNetInput.GetTimeShiftedBlock(-1),
                            Owner.Temporary,
                            Owner.CellsPerBlock,
                            Owner.ForgetGateWeights,
                            Owner.Input,
                            Owner.Input.Count,
                            Owner.Output.GetTimeShiftedBlock(-1),
                            Owner.Output.Count,
                            Owner.CellStates.GetTimeShiftedBlock(-1),
                            1,
                            1
                        );

                        m_netInputFeedForwardKernel.Run(
                            Owner.CellInputNetInput.GetTimeShiftedBlock(-1),
                            Owner.Temporary,
                            Owner.CellsPerBlock,
                            Owner.CellInputWeights,
                            Owner.Input,
                            Owner.Input.Count,
                            Owner.Output.GetTimeShiftedBlock(-1),
                            Owner.Output.Count,
                            nullptr,
                            0,
                            1
                        );

                        m_cellStateFeedForwardKernel.Run(
                            (int)Owner.InputActivationFunction,
                            (int)Owner.GateActivationFunction,

                            Owner.CellStates.GetTimeShiftedBlock(-1),
                            Owner.CellStates.GetTimeShiftedBlock(-1),
                            Owner.CellStateActivations.GetTimeShiftedBlock(-1),
                            Owner.CellStateActivationDerivatives.GetTimeShiftedBlock(-1),

                            Owner.CellInputNetInput.GetTimeShiftedBlock(-1),
                            Owner.CellInputActivations.GetTimeShiftedBlock(-1),
                            Owner.CellInputActivationDerivatives.GetTimeShiftedBlock(-1),

                            Owner.InputGateNetInput.GetTimeShiftedBlock(-1),
                            Owner.InputGateActivations.GetTimeShiftedBlock(-1),
                            Owner.InputGateActivationDerivatives.GetTimeShiftedBlock(-1),

                            Owner.ForgetGateNetInput.GetTimeShiftedBlock(-1),
                            Owner.ForgetGateActivations.GetTimeShiftedBlock(-1),
                            Owner.ForgetGateActivationDerivatives.GetTimeShiftedBlock(-1),

                            Owner.CellStates.Count,
                            Owner.CellsPerBlock,
                            CLIP_CELL_STATE
                        );

                        m_netInputFeedForwardKernel.Run(
                            Owner.OutputGateNetInput.GetTimeShiftedBlock(-1),
                            Owner.Temporary,
                            Owner.CellsPerBlock,
                            Owner.OutputGateWeights,
                            Owner.Input,
                            Owner.Input.Count,
                            Owner.Output.GetTimeShiftedBlock(-1),
                            Owner.Output.Count,
                            Owner.CellStates,
                            1,
                            1
                        );

                        m_outputStateFeedForwardKernel.Run(
                            (int)Owner.GateActivationFunction,

                            Owner.CellStateActivations.GetTimeShiftedBlock(-1),

                            Owner.Output.GetTimeShiftedBlock(-1),
                            Owner.OutputGateNetInput.GetTimeShiftedBlock(-1),
                            Owner.OutputGateActivations.GetTimeShiftedBlock(-1),
                            Owner.OutputGateActivationDerivatives.GetTimeShiftedBlock(-1),

                            Owner.CellStates.Count,
                            Owner.CellsPerBlock,
                            CLIP_CELL_STATE
                        );
                    }

                    m_netInputFeedForwardKernel.Run(
                        Owner.InputGateNetInput,
                        Owner.Temporary,
                        Owner.CellsPerBlock,
                        Owner.InputGateWeights,
                        Owner.Input,
                        Owner.Input.Count,
                        Owner.Output.GetTimeShiftedBlock(-1),
                        Owner.Output.Count,
                        Owner.CellStates.GetTimeShiftedBlock(-1),
                        1,
                        1
                    );

                    m_netInputFeedForwardKernel.Run(
                        Owner.ForgetGateNetInput,
                        Owner.Temporary,
                        Owner.CellsPerBlock,
                        Owner.ForgetGateWeights,
                        Owner.Input,
                        Owner.Input.Count,
                        Owner.Output.GetTimeShiftedBlock(-1),
                        Owner.Output.Count,
                        Owner.CellStates.GetTimeShiftedBlock(-1),
                        1,
                        1
                    );

                    m_netInputFeedForwardKernel.Run(
                        Owner.CellInputNetInput,
                        Owner.Temporary,
                        Owner.CellsPerBlock,
                        Owner.CellInputWeights,
                        Owner.Input,
                        Owner.Input.Count,
                        Owner.Output.GetTimeShiftedBlock(-1),
                        Owner.Output.Count,
                        nullptr,
                        0,
                        1
                    );

                    m_cellStateFeedForwardKernel.Run(
                        (int)Owner.InputActivationFunction,
                        (int)Owner.GateActivationFunction,

                        Owner.CellStates.GetTimeShiftedBlock(-1),
                        Owner.CellStates,
                        Owner.CellStateActivations,
                        Owner.CellStateActivationDerivatives,

                        Owner.CellInputNetInput,
                        Owner.CellInputActivations,
                        Owner.CellInputActivationDerivatives,

                        Owner.InputGateNetInput,
                        Owner.InputGateActivations,
                        Owner.InputGateActivationDerivatives,

                        Owner.ForgetGateNetInput,
                        Owner.ForgetGateActivations,
                        Owner.ForgetGateActivationDerivatives,

                        Owner.CellStates.Count,
                        Owner.CellsPerBlock,
                        CLIP_CELL_STATE
                    );

                    m_netInputFeedForwardKernel.Run(
                        Owner.OutputGateNetInput,
                        Owner.Temporary,
                        Owner.CellsPerBlock,
                        Owner.OutputGateWeights,
                        Owner.Input,
                        Owner.Input.Count,
                        Owner.Output.GetTimeShiftedBlock(-1),
                        Owner.Output.Count,
                        Owner.CellStates,
                        1,
                        1
                    );

                    m_outputStateFeedForwardKernel.Run(
                        (int)Owner.GateActivationFunction,

                        Owner.CellStateActivations,

                        Owner.Output,
                        Owner.OutputGateNetInput,
                        Owner.OutputGateActivations,
                        Owner.OutputGateActivationDerivatives,

                        Owner.CellStates.Count,
                        Owner.CellsPerBlock,
                        CLIP_CELL_STATE
                    ); 
                   
                    break;
                }
            }

            Owner.CellStates.CopyToMemoryBlock(Owner.InnerCellStates, 0, 0, Owner.InnerCellStates.Count);

        }
    }
}

