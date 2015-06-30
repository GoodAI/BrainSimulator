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
    /// <summary>
    /// Empty task that hides unused neural layer tasks.
    /// Doesn't do anything and can be safely ignored.
    /// </summary>
    [Description("EmptyTask"), MyTaskInfo(OneShot = true)]
    public class MyEmptyTask : MyAbstractBackDeltaTask<MyAbstractLayer>
    {
        public MyEmptyTask() { } //parameterless constructor

        public override void Init(int nGPU)
        {
        }

        public override void Execute() //Task execution
        {
        }
    }
}
