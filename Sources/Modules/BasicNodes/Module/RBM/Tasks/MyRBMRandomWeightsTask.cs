using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.RBM.Tasks
{

    /// <summary>
    /// Randomly initalizes the weights of the RBM network using normal distribution and specified standard deviation.
    /// </summary>
    [Description("RBMRandomWeightsTask"), MyTaskInfo(OneShot = true)]
    public class MyRBMRandomWeightsTask : MyTask<MyRBMLayer>
    {
        //Properties
        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = 0ul)]
        public ulong RandomSeed { get; set; }

        [YAXSerializableField(DefaultValue = 0.01f)]
        public float StandardDeviation { get; set; }

        //parameterless constructor
        public MyRBMRandomWeightsTask() { }

        //Kernel initialization
        public override void Init(int nGPU)
        {
        }

        //Task execution
        public override void Execute()
        {
            MyLog.DEBUG.WriteLine("Initialising random weights for " + Owner + " with stdDev: " + StandardDeviation);
            MyKernelFactory.Instance.GetRandDevice(Owner).SetPseudoRandomGeneratorSeed(RandomSeed); // set random seed
            MyKernelFactory.Instance.GetRandDevice(Owner).GenerateNormal(Owner.Weights.GetDevice(Owner), 0, StandardDeviation);
        }
    }
}
