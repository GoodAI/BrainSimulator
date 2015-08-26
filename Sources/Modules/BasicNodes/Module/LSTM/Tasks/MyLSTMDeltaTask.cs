using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using ManagedCuda.BasicTypes;
using YAXLib;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Tasks;
using GoodAI.Core;


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

            // pointer to previous layer
            //MyAbstractLayer previousLayer = Owner.PreviousTopologicalLayer;
            MyAbstractLayer previousLayer = Owner.PreviousLayer;
            CUdeviceptr prevInputPtr = new CUdeviceptr(0); // 2 improve
            if (previousLayer != null)
            {
                // reset delta
                previousLayer.Delta.Fill(0);

                // determine input to previous layer
                if (previousLayer is MyAbstractWeightLayer)
                    prevInputPtr = (previousLayer as MyAbstractWeightLayer).NeuronInput.GetDevicePtr(previousLayer.GPU);
                else
                    prevInputPtr = previousLayer.Input.GetDevicePtr(previousLayer.GPU);
            }

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

                    if (previousLayer != null)
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
                    }
                    break;
                }
                case MyLSTMLayer.LearningTasksType.BPTT:
                {
                    // propagate delta to output gates
                    m_deltaKernel.Run(
                        Owner.Delta,
                        Owner.CellStates,
                        Owner.PreviousCellStates,
                        Owner.CellStateErrors,

                        Owner.OutputGateDeltas,
                        Owner.ForgetGateDeltas,
                        Owner.InputGateDeltas,
                        Owner.CellInputDeltas,

                        Owner.OutputGateActivations,
                        Owner.ForgetGateActivations,
                        Owner.InputGateActivations,

                        Owner.CellInputActivationDerivatives,
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
                        Owner.PreviousOutput,
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
                        Owner.PreviousOutput,

                        Owner.CellInputDeltas,
                        Owner.CellInputWeightGradient,

                        Owner.Input.Count,
                        Owner.CellStates.Count,
                        Owner.CellsPerBlock
                    );

                    if (previousLayer != null)
                    {
                        // propagate delta to previous layer
                        m_deltaBackKernel.SetupExecution(previousLayer.Neurons);
                        m_deltaBackKernel.Run(
                            (int)previousLayer.ActivationFunction,
                            previousLayer.Delta,

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
                    }
                    break;
                }
            }

        }
    }
}