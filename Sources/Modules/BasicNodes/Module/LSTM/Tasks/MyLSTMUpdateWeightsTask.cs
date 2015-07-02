using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using YAXLib;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Core;


namespace GoodAI.Modules.LSTM.Tasks
{
    /// <summary>Updates all network weights according to gradient.</summary>
    [Description("Update weights"), MyTaskInfo(OneShot = false)]
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
            if (SimulationStep == 0) return;

            int backPropMethod;
            float trainingRate, momentum, smoothingFactor;
            if (Owner.ParentNetwork.GetEnabledTask("BackPropagation") is MyRMSTask)
            {
                backPropMethod = 1;
                trainingRate = Owner.ParentNetwork.RMS.TrainingRate;
                momentum = Owner.ParentNetwork.RMS.Momentum;
                smoothingFactor = Owner.ParentNetwork.RMS.SmoothingFactor;
            }
            else
            {
                backPropMethod = 0;
                trainingRate = Owner.ParentNetwork.SGD.TrainingRate;
                momentum = Owner.ParentNetwork.SGD.Momentum;
                smoothingFactor = 0;
            }

            m_updateGateWeightsKernel.Run(
                Owner.Input,
		        Owner.PreviousOutput,
		        Owner.CellStates,
		        Owner.CellStateErrors,
		        Owner.OutputGateDeltas,
		        Owner.InputGateWeights,
                Owner.InputGateWeightDeltas,
                Owner.InputGateWeightMeanSquares,
		        Owner.ForgetGateWeights,
                Owner.ForgetGateWeightDeltas,
                Owner.ForgetGateWeightMeanSquares,
		        Owner.OutputGateWeights,
                Owner.OutputGateWeightDeltas,
                Owner.OutputGateWeightMeanSquares,
		        Owner.InputGateWeightsRTRLPartials,
		        Owner.ForgetGateWeightsRTRLPartials,

                backPropMethod,
                trainingRate,
                momentum,
                smoothingFactor,

		        Owner.Input.Count,
		        Owner.PreviousOutput.Count,
		        Owner.CellsPerBlock
                );

            m_updateCellWeightsKernel.Run(
  		        Owner.Input,
		        Owner.PreviousOutput,
		        Owner.CellStateErrors,
		        Owner.CellInputWeights,
                Owner.CellInputWeightDeltas,
                Owner.CellInputWeightMeanSquares,
		        Owner.CellWeightsRTRLPartials,
               
                backPropMethod,
                trainingRate,
                momentum,
                smoothingFactor,

                Owner.Input.Count,
                Owner.PreviousOutput.Count,
                Owner.CellsPerBlock
                );
        }
    }
}