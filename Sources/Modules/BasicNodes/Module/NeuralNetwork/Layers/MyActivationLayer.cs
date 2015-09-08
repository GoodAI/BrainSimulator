using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;
using YAXLib;

namespace CustomModels.NeuralNetwork.Layers
{

    /// <author>GoodAI</author>
    /// <meta>mz</meta>
    /// <status>WIP</status>
    /// <summary>Activation layer.</summary>
    /// <description>Activation layer of the NN group.</description>
    public class MyActivationLayer : MyAbstractLayer
    {
        public override ConnectionType Connection
        {
            get { return ConnectionType.ONE_TO_ONE; }
        }

        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("\tLayer")]
        public ActivationFunctionType ActivationFunction { get; set; }

        public MyMemoryBlock<float> Delta { get; protected set; }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            if (Neurons > 0)
            {
                // allocate memory scaling with number of neurons in layer
                Delta.Count = Neurons;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator); // commented out, so we don't explicitly need a target input
            validator.AssertError(Input != null, this, "No input available");
            validator.AssertError(Input.Count == Output.Count, this, "Input size must be equal to output size.");
        }

        // description
        public override string Description
        {
            get
            {
                return "Activation layer";
            }
        }
    }



}
