using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using BrainSimulator.Task;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Threading.Tasks;
using YAXLib;
using ManagedCuda;
using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.NeuralNetwork.Tasks;

namespace BrainSimulator.NeuralNetwork.Layers
{

    /// <author>Philip Hilm</author>
    /// <status>OBSOLETE</status>
    /// <summary>Neural layer node.</summary>
    /// <description></description>
    /// 
    /// 
    /// WARNING: THIS LAYER IS NOW OBSOLETE. USE MYHIDDENLAYER / MYOUTPUTLAYER
    public class MyLayer : MyAbstractLayer
    {
        // Properties
        [YAXSerializableField(DefaultValue = ActivationFunctionType.SIGMOID)]
        [MyBrowsable, Category("\tLayer")]
        public ActivationFunctionType ActivationFunction { get; set; }

        [YAXSerializableField(DefaultValue = false)] // target should only be shown when this is true
        [MyBrowsable, Category("\tLayer")]
        public bool ProvideTarget { get; set; }

        // Memory blocks
        [MyInputBlock(1)]
        public MyMemoryBlock<float> Target
        {
            get { return GetInput(1); }
        }

        public override ConnectionType Connection
        {
            get { return ConnectionType.FULLY_CONNECTED;}
        }

        
        public MyMemoryBlock<float> WeightedInput { get; protected set; }

        [MyPersistable]
        public MyMemoryBlock<float> Weights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> PreviousWeightDelta { get; protected set; }

        [MyPersistable]
        public MyMemoryBlock<float> Bias { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> PreviousBiasDelta { get; protected set; }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
            if (Neurons > 0)
            {
                // allocate memory scaling with number of neurons in layer
                WeightedInput.Count = Neurons;
                PreviousBiasDelta.Count = Neurons;
                Bias.Count = Neurons;

                // allocate memory scaling with input
                if (Input != null)
                {
                    PreviousWeightDelta.Count = Neurons * Input.Count;
                    Weights.Count = Neurons * Input.Count;
                }

            }
        }

        // Tasks
        public MyInitLayerTask InitLayer { get; protected set; }
        public MyFeedForwardTask FeedForward { get; protected set; }
        public MyUpdateWeightsTaskDeprecated UpdateWeights { get; protected set; }
        public MyCalcDeltaTask CalcDelta { get; protected set; }
        
        // Parameterless constructor
        public MyLayer()
        {
        }

        //Validation rules
        public override void Validate(MyValidator validator)
        {
            //base.Validate(validator); // commented out, so we don't explicitly need a target input
            //validator.AssertError(Input != null, this, "No input available");
            //validator.AssertError(Neurons > 0, this, "Number of neurons in layer should be > 0");
            //validator.AssertError(Parent is MyNeuralNetworkGroup, this, "A layer needs a parent NeuralNetwork group");
        }

        // description
        public override string Description
        {
            get
            {
                return "Neural layer";
            }
        }
    }

    public enum ActivationFunctionType
    {
        NO_ACTIVATION,
        SIGMOID,
        IDENTITY,
        GAUSSIAN,
        RATIONAL_SIGMOID,
        RELU,
        TANH
    }
}