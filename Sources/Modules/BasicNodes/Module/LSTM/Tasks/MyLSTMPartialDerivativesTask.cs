using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;


namespace GoodAI.Modules.LSTM.Tasks
{
    /// <summary>Computes truncated RTRL partial derivatives.</summary>
    [Description("RTRL Partial Derivatives"), MyTaskInfo(OneShot = false)]
    public class MyLSTMPartialDerivativesTask : MyTask<MyLSTMLayer>
    {
        private MyCudaKernel m_cellWeightsRTRLPartialsKernel;
        private MyCudaKernel m_gateWeightsRTRLPartialsKernel;

        public override void Init(int nGPU)
        {
            m_cellWeightsRTRLPartialsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMPartialDerivativesKernel", "LSTMCellWeightsRTRLPartialsKernel");
            m_gateWeightsRTRLPartialsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMPartialDerivativesKernel", "LSTMGateWeightsRTRLPartialsKernel");

            m_cellWeightsRTRLPartialsKernel.SetupExecution(Owner.CellWeightsRTRLPartials.Count);
            m_gateWeightsRTRLPartialsKernel.SetupExecution(Owner.InputGateWeightsRTRLPartials.Count);
        }

        public override void Execute()
        {
            if (SimulationStep == 0) return;

            m_cellWeightsRTRLPartialsKernel.Run(
                Owner.Input,
		        Owner.PreviousOutput,
		        Owner.InputGateActivations,
		        Owner.ForgetGateActivations,
		        Owner.CellInputActivationDerivatives,
		        Owner.CellWeightsRTRLPartials,

		        Owner.Input.Count,
		        Owner.PreviousOutput.Count,
		        Owner.CellsPerBlock,
                Owner.CellWeightsRTRLPartials.Count
            );
            
            m_gateWeightsRTRLPartialsKernel.Run(
                Owner.Input,
		        Owner.PreviousOutput,
		        Owner.PreviousCellStates,
		        Owner.CellInputActivations,
		        Owner.InputGateActivations,
		        Owner.ForgetGateActivations,
		        Owner.InputGateActivationDerivatives,
		        Owner.ForgetGateActivationDerivatives,
		        Owner.InputGateWeightsRTRLPartials,
		        Owner.ForgetGateWeightsRTRLPartials,

		        Owner.Input.Count,
		        Owner.PreviousOutput.Count,
		        Owner.CellsPerBlock,
                Owner.InputGateWeightsRTRLPartials.Count
            );
        }
    }
}