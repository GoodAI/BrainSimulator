using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>WIP</status>
    /// <summary>Output layer node.</summary>
    /// <description></description>
    public class MyAbstractOutputLayer : MyHiddenLayer
    {
        // Properties
        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tLayer")]
        [ReadOnly(true)]
        public override int Neurons { get; set; }

        // Memory blocks
        public virtual MyMemoryBlock<float> Target { get; protected set; }

        [MyOutputBlock(1)]
        public virtual MyMemoryBlock<float> Cost
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            Cost.Count = 1;
        }

        // Tasks
        [MyTaskGroup("LossFunctions")]
        public MySquaredLossTask SquaredLoss { get; protected set; }
        [MyTaskGroup("LossFunctions")]
        public MyCrossEntropyLossTask CrossEntropyLoss { get; protected set; }
        [MyTaskGroup("LossFunctions")]
        public MyCustomLossTask CustomLoss { get; protected set; }
        // put more loss functions here according to: http://image.diku.dk/shark/sphinx_pages/build/html/rest_sources/tutorials/concepts/library_design/losses.html
        //AbsoluteLoss
        //SquaredLoss
        //ZeroOneLoss
        //DiscreteLoss
        //CrossEntropy
        //CrossEntropyIndependent
        //HingeLoss
        //SquaredHingeLoss
        //EpsilonHingeLoss
        //SquaredEpsilonHingeLoss
        //HuberLoss
        //TukeyBiweightLoss

        // description
        public override string Description
        {
            get
            {
                return "Output layer";
            }
        }

        public void AddRegularization() //Task execution
        {
            if (ParentNetwork.L1 > 0 || ParentNetwork.L2 > 0) // minimize performance hit
            {
                // copy cost to host
                Cost.SafeCopyToHost();

                // sum up L1 and L2 reg terms
                float L1Sum = 0.0f;
                float L2Sum = 0.0f;

                MyAbstractLayer layer = ParentNetwork.FirstTopologicalLayer; // pointer to first layer
                while (layer != null)
                {
                    if (layer is MyAbstractWeightLayer)
                    {
                        // pointer to weight layer
                        MyAbstractWeightLayer weightLayer = layer as MyAbstractWeightLayer;

                        // copy terms to host
                        weightLayer.L1Term.SafeCopyToHost();
                        weightLayer.L2Term.SafeCopyToHost();

                        // add to sums
                        L1Sum += weightLayer.L1Term.Host[0] * ParentNetwork.L1; // TODO: this should be modified if the layer has it's own L1
                        L2Sum += weightLayer.L2Term.Host[0] * ParentNetwork.L2; // TODO: this should be modified if the layer has it's own L2
                    }

                    // next layer
                    layer = layer.NextTopologicalLayer;
                }

                // add sums to cost
                Cost.Host[0] += L1Sum + L2Sum;

                // back to device
                Cost.SafeCopyToDevice();
            }
        }
    }
}
