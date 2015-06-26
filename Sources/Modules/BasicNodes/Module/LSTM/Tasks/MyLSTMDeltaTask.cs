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
    [Description("Calculate deltas"), MyTaskInfo(OneShot = false)]
    public class MyLSTMDeltaTask : MyTask<MyLSTMLayer>
    {
        private MyCudaKernel m_deltaKernel;

        public override void Init(int nGPU)
        {
            m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMDeltaKernel", "LSTMDeltaKernel");
            m_deltaKernel.SetupExecution(Owner.MemoryBlocks);
        }

        public override void Execute()
        {
            MyLayer nextLayer = Owner.NextLayer as MyLayer;

            m_deltaKernel.Run(
                Owner.CellStateErrors,
		        Owner.OutputGateDeltas,
		        Owner.CellStates,
		        Owner.OutputGateActivations,
		        Owner.OutputGateActivationDerivatives,
		        nextLayer.Delta,
		        nextLayer.Weights,

		        nextLayer.Neurons,
		        Owner.CellStates.Count,
		        Owner.CellsPerBlock
                );
        }
    }
}