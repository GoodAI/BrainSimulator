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
    [Description("Update Weights"), MyTaskInfo(OneShot = false)]
    public class MyLSTMUpdateWeightsTask : MyTask<MyLSTMLayer>
    {
        private MyCudaKernel m_updateGateWeightsKernel;
        private MyCudaKernel m_updateCellWeightsKernel;

        public override void Init(int nGPU)
        {
            m_updateGateWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMUpdateWeightsKernel", "LSTMUpdateGateWeightsKernel");
            m_updateCellWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMUpdateWeightsKernel", "LSTMUpdateCellWeightsKernel");

            m_updateGateWeightsKernel.SetupExecution((Owner.Input.Count + Owner.Output.Count + Owner.CellsPerBlock + 1) * Owner.MemoryBlocks);
            m_updateCellWeightsKernel.SetupExecution((Owner.Input.Count + Owner.Output.Count + 1) * Owner.CellStates.Count);
        }

        public override void Execute()
        {
            m_updateGateWeightsKernel.Run(
                Owner.Input,
		        Owner.PreviousOutput,
		        Owner.CellStates,
		        Owner.CellStateErrors,
		        Owner.OutputGateDeltas,
		        Owner.InputGateWeights,
		        Owner.ForgetGateWeights,
		        Owner.OutputGateWeights,
		        Owner.InputGateWeightsRTRLPartials,
		        Owner.ForgetGateWeightsRTRLPartials,

                Owner.LearningRate,
		        Owner.Input.Count,
		        Owner.PreviousOutput.Count,
		        Owner.CellsPerBlock
                );

            m_updateCellWeightsKernel.Run(
  		        Owner.Input,
		        Owner.PreviousOutput,
		        Owner.CellStateErrors,
		        Owner.CellInputWeights,
		        Owner.CellWeightsRTRLPartials,

		        Owner.LearningRate,
                Owner.Input.Count,
                Owner.PreviousOutput.Count,
                Owner.CellsPerBlock
                );
        }
    }
}