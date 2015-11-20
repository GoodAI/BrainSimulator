using GoodAI.Core;
using System.ComponentModel;
using YAXLib;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Core;

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

            Owner.Output.FillAll(0);
            Owner.Delta.FillAll(0);
            Owner.CellStateErrors.FillAll(0);
            Owner.CellInputDeltas.FillAll(0);
            Owner.OutputGateDeltas.FillAll(0);
            Owner.ForgetGateDeltas.FillAll(0);
            Owner.InputGateDeltas.FillAll(0);
            Owner.CellInputWeightGradient.FillAll(0);
            Owner.OutputGateWeightGradient.FillAll(0);
            Owner.InputGateWeightGradient.FillAll(0);
            Owner.ForgetGateWeightGradient.FillAll(0);
            Owner.CellStates.FillAll(0);

            Owner.CellInputActivations.FillAll(0);
            Owner.InputGateActivations.FillAll(0);
            Owner.ForgetGateActivations.FillAll(0);
            Owner.OutputGateActivations.FillAll(0);

            Owner.CellInputActivationDerivatives.FillAll(0);
            Owner.InputGateActivationDerivatives.FillAll(0);
            Owner.ForgetGateActivationDerivatives.FillAll(0);
            Owner.OutputGateActivationDerivatives.FillAll(0);
        }
    }
}
