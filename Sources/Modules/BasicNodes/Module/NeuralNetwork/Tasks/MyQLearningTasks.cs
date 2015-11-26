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

            for (int i = 0; i < Owner.Action.Count; ++i)
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

    /// <summary>Batched Q-Learning. Minibatches must be created from (S_t, S_{t+1}, A_t, R_{t+1}) tuples. 
    /// <ul>
    /// <li>S_t is state of the world in current state</li>
    /// <li>S_{t+1} is the next state</li>
    /// <li>A_t is the action agent did in S_t</li>
    /// <li>R_{t+1} is the reward obtained in next state</li>
    /// </ul> <br />
    /// Input of the network is then S_now, S_t minibatch and S_{t+1} minibatch stacked together. S_now is current world state.<br />
    /// BatchSize of NN group has to be set to 2 * X + 1 where X is BatchSize set at ReplayBuffer node.
    /// </summary>
    [Description("Batched QLearning"), MyTaskInfo(OneShot = false)]
    public class MyQLearningBatchTask : MyTask<MyQLearningLayer>
    {
        // properties
        [YAXSerializableField(DefaultValue = 0.99f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float DiscountFactor { get; set; }

        public MyQLearningBatchTask() { } //parameterless constructor

        public override void Init(int nGPU)
        {

        }

        public override void Execute() //Task execution
        {
            if (SimulationStep == 0)
            {
                Owner.Output.FillAll(0);
            }

            Owner.Output.SafeCopyToHost();

            Owner.Output.CopyToMemoryBlock(Owner.SnowOutput, 0, 0, Owner.SnowOutput.Count);
            Owner.Output.CopyToMemoryBlock(Owner.S0Output, Owner.SnowOutput.Count, 0, Owner.S0Output.Count);
            Owner.Output.CopyToMemoryBlock(Owner.S1Output, Owner.SnowOutput.Count + Owner.S0Output.Count, 0, Owner.S1Output.Count);

            Owner.Action.SafeCopyToHost();
            Owner.S1Output.SafeCopyToHost();
            Owner.Reward.SafeCopyToHost();

            // do not set target for current state
            for (int i = 0; i < Owner.SnowOutput.Count; i++)
            {
                Owner.Target.Host[i] = float.NaN;
            }

            // set target for s0 states

            for (int b = 0; b < (Owner.ParentNetwork.BatchSize - 1) / 2; b++)
            {
                for (int i = 0; i < Owner.Actions; ++i)
                {
                    float target;
                    if (Owner.Action.Host[b * Owner.Actions + i] > 0.5)
                        target = Owner.Reward.Host[b] + DiscountFactor * Owner.S1Output.Host.Skip(b * Owner.Actions).Take(Owner.Actions).Max();
                    else
                        target = float.NaN;

                    Owner.Target.Host[Owner.SnowOutput.Count + b * Owner.Actions + i] = target;
                }
            }

            // do not set target for s1 states
            for (int i = Owner.SnowOutput.Count + Owner.S0Output.Count; i < Owner.Output.Count; i++)
            {
                Owner.Target.Host[i] = float.NaN;
            }

            Owner.Target.SafeCopyToDevice();
        }
    }
}