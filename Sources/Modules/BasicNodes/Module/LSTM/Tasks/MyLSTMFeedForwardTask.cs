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
        [YAXSerializableField(DefaultValue = 0f)]
        [MyBrowsable, Category("\tLayer")]
        public float CLIP_CELL_STATE { get; set; }

        private MyCudaKernel m_feedForwardKernel;

        public MyLSTMFeedForwardTask() { }

        public override void Init(int nGPU)
        {
            m_feedForwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMFeedForwardKernel", "LSTMFeedForwardKernel");
            m_feedForwardKernel.SetupExecution(Owner.MemoryBlocks);
        }

        public override void Execute()
        {
            if (Owner.ResetSignal.IsIncomingRised())
            {
                Owner.CellStates.Fill(0);
                Owner.Output.Fill(0);
            }

            Owner.CellStates.CopyToMemoryBlock(Owner.PreviousCellStates, 0, 0, Owner.CellStates.Count);
            Owner.Output.CopyToMemoryBlock(Owner.PreviousOutput, 0, 0, Owner.Output.Count);

            m_feedForwardKernel.Run(
                (int) Owner.InputActivationFunction,
                (int) Owner.GateActivationFunction,
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
        }
    }
}

