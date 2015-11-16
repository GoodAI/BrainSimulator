using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>This is the main QLearning algorithm, that uses the current timestep as 't+1' and the previous timestep as 't'
    /// </summary>
    /// <description></description>
    [Description("QLearning"), MyTaskInfo(OneShot = false)]
    public class MyQLearningTask : MyTask<MyQLearningLayer>
    {
        // properties
        [YAXSerializableField(DefaultValue = 0.99f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float DiscountFactor { get; set; }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float BindLower { get; set; }

        [YAXSerializableField(DefaultValue = 1.00f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float BindUpper { get; set; }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tHyperParameters")]
        public bool BindTarget { get; set; }

        public MyQLearningTask() { } //parameterless constructor

        public override void Init(int nGPU)
        {

        }

        public override void Execute() //Task execution
        {
            if (SimulationStep == 0)
            {
                Owner.Output.FillAll(0);
            }

            Owner.ParentNetwork.FirstTopologicalLayer.Input.CopyToMemoryBlock(Owner.TempInput, 0, 0, Owner.ParentNetwork.FirstTopologicalLayer.Input.Count);
            Owner.Output.SafeCopyToHost();
            Owner.Action.SafeCopyToHost();
            Owner.Reward.SafeCopyToHost();
            Owner.Target.SafeCopyToHost();
            Owner.PreviousOutput.SafeCopyToHost();

            float reward = Owner.Reward.Host[0];
            float gamma = DiscountFactor;

            for (int i = 0; i < Owner.Output.Count; ++i)
            {
                float target;
                if (Owner.Action.Host[i] > 0.5)
                    target = reward + gamma * Owner.Output.Host.Max();
                else
                    target = Owner.PreviousOutput.Host[i];

                if (BindTarget)
                {
                    if (target > BindUpper)
                        target = BindUpper;
                    else if (target < BindLower)
                        target = BindLower;
                }

                Owner.Target.Host[i] = target;
            }
            Owner.Target.SafeCopyToDevice();

            Owner.Output.CopyToMemoryBlock(Owner.PreviousOutput, 0, 0, Owner.Output.Count);


            Owner.PreviousInput.CopyToMemoryBlock(Owner.ParentNetwork.FirstTopologicalLayer.Input, 0, 0, Owner.PreviousInput.Count);
            Owner.TempInput.CopyToMemoryBlock(Owner.PreviousInput, 0, 0, Owner.TempInput.Count);
            Owner.ParentNetwork.FeedForward();
        }
    }
}