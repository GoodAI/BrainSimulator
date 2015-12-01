using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>QLearning output layer node.</summary>
    /// <description>
    /// This node implements QLearning as described on Wikipedia: <a href="https://en.wikipedia.org/wiki/Q-learning"> https://en.wikipedia.org/wiki/Q-learning </a><br></br>
    ///  Number of actions should be set as a parameter<br></br>
    /// As inputs it takes:<br></br>
    ///  - The current state fed through one or more hidden layers<br></br>
    ///  - Reward for the current state<br></br>
    ///  - The action chosen for the previous state as a vector of actions eg. [0, 0, 1] is the last of 3 actions<br></br>
    ///  The output is the estimated value of each action
    /// </description>
    public class MyQLearningLayer : MyAbstractOutputLayer
    {
        // Properties
        [YAXSerializableField(DefaultValue = 3)]
        [MyBrowsable, Category("\tLayer")]
        public int Actions { get; set; }

        public int StateSize;

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

        public MyMemoryBlock<float> PreviousInput { get; protected set; }
        public MyMemoryBlock<float> PreviousOutput { get; protected set; }
        public MyMemoryBlock<float> TempInput { get; protected set; }

        public MyMemoryBlock<float> S0Output { get; protected set; }
        public MyMemoryBlock<float> S1Output { get; protected set; }
        public MyMemoryBlock<float> SnowOutput { get; protected set; }

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

            Target.Count = Neurons * ParentNetwork.BatchSize;
            if (firstLayer.Input != null)
                PreviousInput.Count = TempInput.Count = firstLayer.Input.Count;
            PreviousOutput.Count = Output.Count;

            if (ParentNetwork.BatchSize >= 3)
            {
                StateSize = Output.Count / ParentNetwork.BatchSize;

                S0Output.Count = (ParentNetwork.BatchSize - 1) / 2 * StateSize;
                S1Output.Count = (ParentNetwork.BatchSize - 1) / 2 * StateSize;
                SnowOutput.Count = StateSize;
            }
        }

        // Tasks
        [MyTaskGroup("QLearning")]
        public MyQLearningTask QLearning { get; protected set; }
        [MyTaskGroup("QLearning")]
        public MyQLearningBatchTask QLearningBatch { get; protected set; }

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
            if (QLearning.Enabled)
            {
                validator.AssertError(Action.Count == Neurons, this, "Number of neurons need to correspond with number of actions (action size)");
                validator.AssertError(Reward.Count == 1, this, "Reward needs to be a single floating point number (cannot be an array)");
            } 
            else if (QLearningBatch.Enabled)
            {
                validator.AssertError(ParentNetwork.BatchSize >= 3, this, "BatchSize needs to be >= 3");
                validator.AssertError(Reward.Count == (ParentNetwork.BatchSize - 1) / 2, this, "Reward size must be equal to (BatchSize - 1) / 2");
                validator.AssertError(Action.Count == (ParentNetwork.BatchSize - 1) / 2 * Actions, this, "Action size must be equal to (BatchSize - 1) / 2 * Actions");
            }
        }
    }
}
