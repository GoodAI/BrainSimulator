using GoodAI.Core;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System.ComponentModel;
using YAXLib;


namespace GoodAI.Modules.LSTM.Tasks
{
    /// <summary>Updates all network weights according to gradient. <br />
    /// Parameters:
    /// <ul>
    ///     <li>CLIP_GRADIENT: Limits error gradient into [-CLIP_GRADIENT,CLIP_GRADIENT] interval. Set to 0 for no bounds</li>
    /// </ul>
    /// </summary>
    [Description("Update weights"), MyTaskInfo(OneShot = false)]
    public class MyLSTMUpdateWeightsTask : MyAbstractUpdateWeightsTask<MyLSTMLayer>
    {
        [YAXSerializableField(DefaultValue = 0f)]
        [MyBrowsable, Category("\tLayer")]
        public float CLIP_GRADIENT { get; set; }

        private MyCudaKernel m_updateGateWeightsKernel;
        private MyCudaKernel m_updateCellWeightsKernel;

        public override void Init(int nGPU)
        {
            switch (Owner.LearningTasks)
            {
                case MyLSTMLayer.LearningTasksType.RTRL:
                {
                    m_updateGateWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMUpdateWeightsKernel", "LSTMUpdateGateWeightsKernel");
                    m_updateCellWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMUpdateWeightsKernel", "LSTMUpdateCellWeightsKernel");
                    break;
                }
                case MyLSTMLayer.LearningTasksType.BPTT:
                {
                    m_updateGateWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMUpdateWeightsKernel", "LSTMUpdateGateWeightsKernelBPTT");
                    m_updateCellWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"LSTM\LSTMUpdateWeightsKernel", "LSTMUpdateCellWeightsKernelBPTT");
                    break;
                }
            }
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

            switch (Owner.LearningTasks)
            {
                case MyLSTMLayer.LearningTasksType.RTRL:
                {
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
                        CLIP_GRADIENT,

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
                        CLIP_GRADIENT,

                        Owner.Input.Count,
                        Owner.PreviousOutput.Count,
                        Owner.CellsPerBlock
                    );
                    break;
                }
                case MyLSTMLayer.LearningTasksType.BPTT:
                {
                    m_updateGateWeightsKernel.Run(
                        Owner.InputGateWeights,
                        Owner.InputGateWeightDeltas, // RMS
                        Owner.InputGateWeightMeanSquares, // RMS
                        Owner.ForgetGateWeights,
                        Owner.ForgetGateWeightDeltas, // RMS
                        Owner.ForgetGateWeightMeanSquares, // RMS
                        Owner.OutputGateWeights,
                        Owner.OutputGateWeightDeltas, // RMS
                        Owner.OutputGateWeightMeanSquares, //RMS

                        Owner.OutputGateWeightGradient,
                        Owner.InputGateWeightGradient,
                        Owner.ForgetGateWeightGradient,

                        backPropMethod,
                        trainingRate,
                        momentum,
                        smoothingFactor,
                        CLIP_GRADIENT,

                        Owner.Input.Count,
                        Owner.PreviousOutput.Count,
                        Owner.CellsPerBlock
                    );

                    m_updateCellWeightsKernel.Run(
                        Owner.CellInputWeights,
                        Owner.CellInputWeightDeltas,
                        Owner.CellInputWeightMeanSquares,

                        backPropMethod,
                        trainingRate,
                        momentum,
                        smoothingFactor,
                        CLIP_GRADIENT,

                        Owner.CellInputWeightGradient,

                        Owner.Input.Count,
                        Owner.PreviousOutput.Count
                    );
                    break;
                }
            }

        }
    }
}