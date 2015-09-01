using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.LSTM.Tasks
{
    /// <summary>Initialises layer with random weights.</summary>
    [Description("Init LSTM layer"), MyTaskInfo(OneShot = true)]
    public class MyLSTMInitLayerTask : MyTask<MyLSTMLayer>
    {
        [YAXSerializableField(DefaultValue = 0.25f)]
        [MyBrowsable, Category("\tLayer")]
        public float INIT_WEIGHTS_STDDEV { get; set; }

        public override void Init(int nGPU)
        {
            
        }

        public override void Execute()
        {
            Owner.CellStates.Fill(0);
            Owner.PreviousCellStates.Fill(0);

            Owner.Output.Fill(0);
            Owner.PreviousOutput.Fill(0);

            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.CellInputWeights.GetDevice(Owner), 0, INIT_WEIGHTS_STDDEV);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.InputGateWeights.GetDevice(Owner), 0, INIT_WEIGHTS_STDDEV);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.ForgetGateWeights.GetDevice(Owner), 0, INIT_WEIGHTS_STDDEV);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.OutputGateWeights.GetDevice(Owner), 0, INIT_WEIGHTS_STDDEV);

            Owner.CellWeightsRTRLPartials.Fill(0);
            Owner.InputGateWeightsRTRLPartials.Fill(0);
            Owner.ForgetGateWeightsRTRLPartials.Fill(0);

            Owner.CellInputWeightDeltas.Fill(0);
            Owner.InputGateWeightDeltas.Fill(0);
            Owner.ForgetGateWeightDeltas.Fill(0);
            Owner.OutputGateWeightDeltas.Fill(0);

            Owner.CellInputWeightMeanSquares.Fill(0);
            Owner.InputGateWeightMeanSquares.Fill(0);
            Owner.ForgetGateWeightMeanSquares.Fill(0);
            Owner.OutputGateWeightMeanSquares.Fill(0);
        }
    }
}
