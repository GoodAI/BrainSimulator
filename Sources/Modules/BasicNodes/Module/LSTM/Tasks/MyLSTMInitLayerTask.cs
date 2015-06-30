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
    /// <summary>Initialises layer with random weights.</summary>
    [Description("Init LSTM layer"), MyTaskInfo(OneShot = true)]
    public class MyLSTMInitLayerTask : MyTask<MyLSTMLayer>
    {
        public override void Init(int nGPU)
        {
            
        }

        public override void Execute()
        {
            Owner.CellStates.Fill(0);
            Owner.PreviousCellStates.Fill(0);

            Owner.Output.Fill(0);
            Owner.PreviousOutput.Fill(0);

            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.CellInputWeights.GetDevice(Owner), 0, 0.25f);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.InputGateWeights.GetDevice(Owner), 0, 0.25f);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.ForgetGateWeights.GetDevice(Owner), 0, 0.25f);
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.OutputGateWeights.GetDevice(Owner), 0, 0.25f);

            Owner.CellWeightsRTRLPartials.Fill(0);
            Owner.InputGateWeightsRTRLPartials.Fill(0);
            Owner.ForgetGateWeightsRTRLPartials.Fill(0);

            Owner.CellInputWeightDeltas.Fill(0);
            Owner.InputGateWeightDeltas.Fill(0);
            Owner.ForgetGateWeightDeltas.Fill(0);
            Owner.OutputGateWeightDeltas.Fill(0);
        }
    }
}
