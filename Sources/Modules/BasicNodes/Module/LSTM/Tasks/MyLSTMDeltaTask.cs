using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Tasks;
using ManagedCuda.BasicTypes;
using System.ComponentModel;


namespace GoodAI.Modules.LSTM.Tasks
{
    /// <summary>Computes deltas of output gates and cell state errors.</summary>
    [Description("Calculate deltas"), MyTaskInfo(OneShot = false)]
    public class MyLSTMDeltaTask : MyAbstractBackDeltaTask<MyLSTMLayer>
    {
        public MyLSTMDeltaTask() { }

        private MyCudaKernel m_deltaKernel;
        private MyCudaKernel m_gateGradientKernel;
        private MyCudaKernel m_cellInputGradientKernel;
        private MyCudaKernel m_deltaBackKernel;

        private CUdeviceptr nullCUdeviceptr = new CUdeviceptr(0);

        public override void Init(int nGPU)
        {
            switch (Owner.LearningTasks)
            {
                case MyLSTMLayer.LearningTasksType.RTRL:
                {
                    m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMDeltaKernel");
                    m_deltaBackKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMDeltaBackKernel");
                    break;
                }
                case MyLSTMLayer.LearningTasksType.BPTT:
                {
                    m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMDeltaKernelBPTT");
                    m_gateGradientKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMGateGradientKernelBPTT");
                    m_cellInputGradientKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMCellInputGradientKernelBPTT");
                    m_deltaBackKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMDeltaBackKernelBPTT");

                    m_gateGradientKernel.SetupExecution((Owner.Input.Count + Owner.Output.Count + Owner.CellsPerBlock + 1) * Owner.MemoryBlocks);
                    m_cellInputGradientKernel.SetupExecution((Owner.Input.Count + Owner.Output.Count + 1) * Owner.CellStates.Count);
                    break;
                }
            }
            m_deltaKernel.SetupExecution(Owner.MemoryBlocks);
        }

        public override void Execute()
        {
            if (SimulationStep == 0) return;

            switch (Owner.LearningTasks)
            {
                case MyLSTMLayer.LearningTasksType.RTRL:
                {
                    // propagate delta to output gates
                    m_deltaKernel.Run(
                        Owner.CellStateErrors,
                        Owner.OutputGateDeltas,
                        Owner.CellStates,
                        Owner.OutputGateActivations,
                        Owner.OutputGateActivationDerivatives,
                        Owner.Delta,

                        Owner.CellStates.Count,
                        Owner.CellsPerBlock
                    );
                    break;
                }
                case MyLSTMLayer.LearningTasksType.BPTT:
                {
                    // propagate delta to output gates
                    m_deltaKernel.Run(
                        Owner.Delta,
                        Owner.CellStates,
                        Owner.CellStates.GetTimeShiftedBlock(-1),
                        Owner.CellStateErrors,
                        Owner.CellStateErrors.GetTimeShiftedBlock(+1),

                        Owner.OutputGateDeltas,
                        Owner.ForgetGateDeltas,
                        Owner.ForgetGateDeltas.GetTimeShiftedBlock(+1),
                        Owner.InputGateDeltas,
                        Owner.InputGateDeltas.GetTimeShiftedBlock(+1),
                        Owner.CellInputDeltas,

                        Owner.CellInputActivations,
                        Owner.CellStateActivations,
                        Owner.OutputGateActivations,
                        Owner.ForgetGateActivations.GetTimeShiftedBlock(+1),
                        Owner.InputGateActivations,

                        Owner.CellInputActivationDerivatives,
                        Owner.CellStateActivationDerivatives,
                        Owner.OutputGateActivationDerivatives,
                        Owner.ForgetGateActivationDerivatives,
                        Owner.InputGateActivationDerivatives,

                        Owner.CellInputWeights,
                        Owner.OutputGateWeights,
                        Owner.ForgetGateWeights,
                        Owner.InputGateWeights,

                        Owner.Input.Count,
                        Owner.CellStates.Count,
                        Owner.CellsPerBlock
                    );

                    m_gateGradientKernel.Run(
                        Owner.Input,
                        Owner.Output.GetTimeShiftedBlock(-1),
                        Owner.CellStates,

                        Owner.InputGateDeltas,
                        Owner.ForgetGateDeltas,
                        Owner.OutputGateDeltas,

                        Owner.OutputGateWeightGradient,
                        Owner.InputGateWeightGradient,
                        Owner.ForgetGateWeightGradient,

                        Owner.Input.Count,
                        Owner.CellStates.Count,
                        Owner.CellsPerBlock
                    );

                    m_cellInputGradientKernel.Run(
                        Owner.Input,
                        Owner.Output.GetTimeShiftedBlock(-1),

                        Owner.CellInputDeltas,
                        Owner.CellInputWeightGradient,

                        Owner.Input.Count,
                        Owner.CellStates.Count,
                        Owner.CellsPerBlock
                    );
                    break;
                }
            }

            MyNode node = Owner.GetInput(0).Owner;

            if (node is MyAbstractLayer)
            {
                MyAbstractLayer previousLayer = node as MyAbstractLayer;

                CUdeviceptr prevInputPtr = nullCUdeviceptr;

                // reset delta
                if (Owner.ParentNetwork.TimeStep == 0)
                {
                    previousLayer.Delta.Fill(0);
                }

                // determine input to previous layer
                prevInputPtr = MyAbstractLayer.DetermineInput(previousLayer);

                switch (Owner.LearningTasks)
                {
                    case MyLSTMLayer.LearningTasksType.RTRL:
                    {
                        // propagate delta to previous layer
                        m_deltaBackKernel.SetupExecution(previousLayer.Neurons);
                        m_deltaBackKernel.Run(
                            (int)previousLayer.ActivationFunction,
                            prevInputPtr,
                            previousLayer.Delta,
                            Owner.CellStateErrors,
                            Owner.PreviousCellStates,
                            Owner.InputGateActivations,
                            Owner.CellInputActivationDerivatives,
                            Owner.InputGateActivationDerivatives,
                            Owner.ForgetGateActivationDerivatives,
                            Owner.CellInputWeights,
                            Owner.InputGateWeights,
                            Owner.ForgetGateWeights,
                            Owner.OutputGateWeights,
                            Owner.OutputGateDeltas,

                            previousLayer.Neurons,
                            Owner.CellStates.Count,
                            Owner.CellsPerBlock
                        );
                        break;
                    }
                    case MyLSTMLayer.LearningTasksType.BPTT:
                    {
                        // propagate delta to previous layer
                        m_deltaBackKernel.SetupExecution(previousLayer.Neurons);
                        m_deltaBackKernel.Run(
                            (int)previousLayer.ActivationFunction,
                            prevInputPtr,
                            previousLayer.Delta,

                            Owner.CellInputDeltas,
                            Owner.OutputGateDeltas,
                            Owner.ForgetGateDeltas,
                            Owner.InputGateDeltas,

                            Owner.CellInputWeights,
                            Owner.InputGateWeights,
                            Owner.ForgetGateWeights,
                            Owner.OutputGateWeights,

                            previousLayer.Neurons,
                            Owner.CellStates.Count,
                            Owner.CellsPerBlock
                        );
                        break;
                    }
                }
            }
        }
    }
}
