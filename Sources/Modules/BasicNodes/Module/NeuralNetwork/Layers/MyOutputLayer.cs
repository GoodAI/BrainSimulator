using GoodAI.Core.Memory;
using GoodAI.Core.Utils;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Output layer node.</summary>
    /// <description>
    /// The output layer takes a target as input, and automatically scales it's neurons to fit the target.<br></br>
    /// The cost can be observed or manipulated as an output
    /// </description>
    public class MyOutputLayer : MyAbstractOutputLayer
    {
        // Memory blocks
        [MyInputBlock(1)]
        public override MyMemoryBlock<float> Target
        {
            get { return GetInput(1); }
        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            // automatically set number of neurons to the same size as target
            if (Target != null)
                Neurons = Target.Count / ParentNetwork.BatchSize;

            base.UpdateMemoryBlocks(); // call after number of neurons are set
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Target != null, this, "Target of output layer \"" + this.Name + "\" is not defined.");
        }
    }
}
