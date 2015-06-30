using BrainSimulator.Memory;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using BrainSimulator.NeuralNetwork.Tasks;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.NeuralNetwork.Layers
{
    /// <author>Philip Hilm</author>
    /// <status>Working</status>
    /// <summary>QLearning output layer node.</summary>
    /// <description>
    /// This node implements QLearning as described on Wikipedia: <a href="https://en.wikipedia.org/wiki/Q-learning"> https://en.wikipedia.org/wiki/Q-learning </a><br></br>
    ///  Number of actions should be set as a parameter<br></br>
    /// As inputs it takes:<br></br>
    ///  - The current state fed through one or more hidden layers<br></br>
    ///  - Reward for the current state<br></br>
    ///  - The action chosen for the current state as a vector of actions eg. [0, 0, 1] is the last of 3 actions<br></br>
    ///  The output is the estimated value of each action
    /// </description>
    public class MyQLearningLayer : MyAbstractOutputLayer
    {
        // Properties
        [YAXSerializableField(DefaultValue = 3)]
        [MyBrowsable, Category("\tLayer")]
        public int Actions { get; set; }

        #region Memory blocks
        // Memory blocks
        [MyInputBlock(1)]
        public MyMemoryBlock<float> Reward
        {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> Action
        {
            get { return GetInput(2); }
        }
        public MyMemoryBlock<float> PreviousAction { get; protected set; }
        public MyMemoryBlock<float> PreviousInput { get; protected set; }
        public MyMemoryBlock<float> PreviousReward { get; protected set; }
        public override MyMemoryBlock<float> Target { get; protected set; }
        #endregion

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            Neurons = Actions;

            base.UpdateMemoryBlocks();

            // find first layer
            MyAbstractLayer firstLayer = this;
            while (firstLayer.Input != null && firstLayer.Input.Owner is MyAbstractLayer)
                firstLayer = firstLayer.Input.Owner as MyAbstractLayer;

            Target.Count = Neurons;
            PreviousAction.Count = Neurons;
            if (firstLayer.Input != null)
                PreviousInput.Count = firstLayer.Input.Count;
            PreviousReward.Count = 1;
        }

        // Tasks
        public MyQLearningTask QLearning { get; protected set; }
        public MyRestoreValuesTask RestoreValues { get; protected set; }

        // description
        public override string Description
        {
            get
            {
                return "Q Learning output";
            }
        }

        //Validation rules
        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Action.Count == Neurons, this, "Number of neurons need to correspond with number of actions (action size)");
            validator.AssertError(Reward.Count == 1, this, "Reward needs to be a single floating point number (cannot be an array)");
        }
    }
}
