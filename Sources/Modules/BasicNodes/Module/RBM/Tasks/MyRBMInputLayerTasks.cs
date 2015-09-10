using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using GoodAI.Modules.RBM;
using System.ComponentModel;

namespace CustomModels.RBM.Tasks
{

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// <p>
    /// Simple input forward tasks that copies input to output.
    /// </p>
    /// <br></br>
    /// <p>
    /// Does not have to be checked/used when RBM task is active.
    /// MUST be checked/used when the group is doing an SGD-type task.
    /// </p>
    /// </summary>
    /// <description></description>
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

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>Working</status>
    /// <summary>
    /// <p>
    /// Simple delta backprop tasks that copies output deltas to input deltas.
    /// </p>
    /// <br></br>
    /// <p>
    /// Does not have to be checked/used when RBM task is active.
    /// </p>
    /// </summary>
    /// <description></description>
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
