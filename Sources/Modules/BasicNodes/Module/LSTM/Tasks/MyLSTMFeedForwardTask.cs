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


namespace BrainSimulator.LSTM.Tasks
{
    [Description("Feed forward"), MyTaskInfo(OneShot = false)]
    public class MyLSTMFeedForwardTask : MyTask<MyLSTMLayer>
    {
        private MyCudaKernel m_feedForwardKernel;

        public override void Init(int nGPU)
        {
            m_feedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "LSTMFeedForwardKernel");
            m_feedForwardKernel.SetupExecution(Owner.MemoryBlocks);
        }

        public override void Execute()
        {
            Owner.CellStates.CopyToMemoryBlock(Owner.PreviousCellStates, 0, 0, Owner.CellStates.Count);
            Owner.Output.CopyToMemoryBlock(Owner.PreviousOutput, 0, 0, Owner.Output.Count);

            m_feedForwardKernel.Run(
                (int) Owner.ActivationFunction,
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

                Owner.Input.Count,
                Owner.CellStates.Count,
                Owner.CellsPerBlock
                );
        }
    }
}

