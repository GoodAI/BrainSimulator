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
    /// <status>WIP</status>
    /// <summary>Output layer node.</summary>
    /// <description></description>
    public class MyOutputLayer : MyHiddenLayer
    {
        // Properties
        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tLayer")]
        [ReadOnly(true)]
        public override int Neurons { get; set; }

        // Memory blocks
        [MyInputBlock(1)]
        public MyMemoryBlock<float> Target
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Cost
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        public MyMemoryBlock<float> LastCost { get; protected set; }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            // automatically set number of neurons to the same size as target
            if (Target != null)
                Neurons = Target.Count;

            Cost.Count = 1;
            LastCost.Count = 1;

            base.UpdateMemoryBlocks();
        }

        // Tasks
        [MyTaskGroup("LossFunctions")]
        public MySquaredLossTask SquaredLoss { get; protected set; }
        //[MyTaskGroup("LossFunctions")]
        //public MyCrossEntropyDeltaTask CrossEntropy { get; protected set; }
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

                MyAbstractLayer layer = ParentNetwork.FirstLayer; // pointer to first layer
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
                    layer = layer.NextLayer;
                }

                // add sums to cost
                Cost.Host[0] += L1Sum + L2Sum;

                // back to device
                Cost.SafeCopyToDevice();
            }
        }
    }
}
