using BrainSimulator;
using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.NeuralNetwork.Layers;
using BrainSimulator.RBM;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.NeuralNetwork.Tasks
{
    /// <author>Philip Hilm</author>
    /// <status>Working</status>
    /// <summary>Uses the current and the previous timestep to update the target according to:
    /// <br></br>
    /// <a href="https://en.wikipedia.org/wiki/Q-learning"> https://en.wikipedia.org/wiki/Q-learning </a>
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
        public bool BindValues { get; set; }

        public MyQLearningTask() { } //parameterless constructor

        public override void Init(int nGPU) { }

        public override void Execute() //Task execution
        {
            // backup outputs in host memory and find the best possible value
            Owner.Output.SafeCopyToHost();
            float maxValue = Owner.Output.Host.Max();

            // backup network inputs in host memory
            Owner.ParentNetwork.FirstLayer.Input.SafeCopyToHost();

            // do a forward pass with the previous inputs
            Owner.ParentNetwork.FirstLayer.Input.CopyFromMemoryBlock(Owner.PreviousInput, 0, 0, Owner.PreviousInput.Count);
            Owner.ParentNetwork.FeedForward();

            // set the target: q(s(t-1), a) -> reward(t-1) + q(s(t), maxarg(a)) * discountfactor // TODO: consider reward(t-1) vs reward(t) - reward(t-1) is correct, but reward(t) works better with our implementation of breakout
            Owner.Target.CopyFromMemoryBlock(Owner.Output, 0, 0, Owner.Neurons);
            Owner.Target.SafeCopyToHost(); // manipulate at host
            Owner.PreviousAction.SafeCopyToHost(); // manipulate at host
            Owner.PreviousReward.SafeCopyToHost(); // manipulate at host
            //float value = Owner.PreviousReward.Host[0] + maxValue;
            Owner.Reward.SafeCopyToHost();
            float normalize = 0;
            for (int a = 0; a < Owner.Neurons; a++)
                normalize += Owner.PreviousAction.Host[a];
            if (normalize > 0)
            {
                for (int a = 0; a < Owner.Neurons; a++)
                    Owner.PreviousAction.Host[a] /= normalize;
            }
            else
            {
                for (int a = 0; a < Owner.Neurons; a++)
                    Owner.PreviousAction.Host[a] = 1.0f / Owner.Neurons;
            }
            float value = Owner.Reward.Host[0] + maxValue;
            if (BindValues)
            {
                if (value > BindUpper)
                    value = BindUpper;
                else if (value < BindLower)
                    value = BindLower;
            }

            for (int a = 0; a < Owner.Neurons; a++)
                Owner.Target.Host[a] = Owner.PreviousAction.Host[a] * (value * DiscountFactor) + (1 - Owner.PreviousAction.Host[a]) * Owner.Target.Host[a];

            Owner.Target.SafeCopyToDevice(); // back to device

            // copy current values to previous values
            Owner.PreviousReward.CopyFromMemoryBlock(Owner.Reward, 0, 0, 1);
            Owner.PreviousAction.CopyFromMemoryBlock(Owner.Action, 0, 0, Owner.Neurons);
        }
    }

    /// <author>Philip Hilm</author>
    /// <status>Working</status>
    /// <summary>The QLearning task sets and updates the values from the previous timestep.
    /// <br></br>
    /// This restores the values to the current timestep.
    /// </summary>
    /// <description></description>
    [Description("Restore values"), MyTaskInfo(OneShot = false)]
    public class MyRestoreValuesTask : MyTask<MyQLearningLayer>
    {
        public MyRestoreValuesTask() { } //parameterless constructor

        public override void Init(int nGPU) { }

        public override void Execute() //Task execution
        {
            // restore output values from host
            Owner.Output.SafeCopyToDevice();

            // restore input values from host
            Owner.ParentNetwork.FirstLayer.Input.SafeCopyToDevice();
            Owner.PreviousInput.CopyFromMemoryBlock(Owner.ParentNetwork.FirstLayer.Input, 0, 0, Owner.PreviousInput.Count);
        }
    }
}
