using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Tasks;
using GoodAI.Modules.RBM;
using GoodAI.Core.Utils;
using ManagedCuda.BasicTypes;

namespace CustomModels.RBM.Tasks
{

    [Description("RBMInputForward"), MyTaskInfo(OneShot = false)]
    public class MyRBMInputForwardTask : MyAbstractForwardTask<MyRBMInputLayer>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.Output.CopyFromMemoryBlock(Owner.Input, 0, 0, Owner.Neurons);
            MyLog.DEBUG.WriteLine("RBM input layer forward");
        }
    }

    [Description("RBMInputBackward"), MyTaskInfo(OneShot = false)]
    public class MyRBMInputBackwardTask : MyAbstractBackDeltaTask<MyRBMInputLayer>
    {
        public override void Init(int nGPU)
        {
        }

        public override void Execute()
        {
            Owner.Input.CopyFromMemoryBlock(Owner.Output, 0, 0, Owner.Neurons);
            MyLog.DEBUG.WriteLine("RBM input layer backwards");
        }
    }
}
