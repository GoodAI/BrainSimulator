using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Memory;
using GoodAI.Core.Utils;
using GoodAI.Core.Task;
using GoodAI.Core.Signals;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Threading.Tasks;
using YAXLib;
using ManagedCuda;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Tasks;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    /// <author>GoodAI</author>
    /// <meta>mbr</meta>
    /// <status>Development</status>
    /// <summary>Hidden gaussian layer.</summary>
    /// <description>
    /// This is hidden layer but each pair of neurons is interpreted as parameters of Gaussian distribution.
    /// Parameter 'Neurons' (or N) sets the number of Gaussians where each Gaussian is represented by two neurons.
    /// Format is like this: input -> {mu_1, sigma_1,..., mu_2*N, sigma_2*N} -> {output_1, ..., output_N}.
    /// Backprop has regularization term implementing Variational Bayes.
    /// </description>
    public class MyGaussianHiddenLayer : MyAbstractWeightLayer, IMyCustomTaskFactory
    {
        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tSigma")]
        // Wheter to use constant sigma or not
        public bool UseSigmaConstant { get; set; }

        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("\tSigma")]
        // If UseSigmaConstant = true, sigmas will be set to this value
        // It also sets neurons count = input count, and use all neurons from prev layer as means
        public float SigmaConstant { get; set; }

        [MyPersistable]
        // If Generate is on, it samples input from prior and does not propagate delta
        public MyGenerateSignal Generate { get; protected set; }
        public class MyGenerateSignal : MySignal { }

        // Samples from random uniform distriution used for sampling in generative mode
        public MyMemoryBlock<float> RandomUniform { get; protected set; }

        // Those are transformed by: NoisyInput = mean + RandomNormal * sigma
        public MyMemoryBlock<float> RandomNormal { get; protected set; }
        public MyMemoryBlock<float> NoisyInput { get; protected set; }

        // Regularization loss is stored here (computation is turned off by default)
        public MyMemoryBlock<float> Regularization { get; protected set; }

        // Mins and maxes of both means and sigmas so far, can be reset
        public MyMemoryBlock<float> PriorGaussHiddenStatesMin { get; protected set; }
        public MyMemoryBlock<float> PriorGaussHiddenStatesMax { get; protected set; }

        // If UseSigmaConstant is true then this will be the block filled with SigmaConstant
        public MyMemoryBlock<float> SigmaConstants { get; protected set; }

        public override ConnectionType Connection
        {
            get { return ConnectionType.GAUSSIAN; }
        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            if (UseSigmaConstant)
            {
                // Use whole layer for means
                Neurons = Input != null ? Input.Count : 1;
                // Allocate memory for constants
                SigmaConstants.Count = Neurons;
            }
            else
            {
                // Both means and sigmas are in previous layer
                // Two neurons (mean, sigma) from previous layer for each neuron in this layer
                Neurons = Input != null ? Input.Count / 2 : 1;
                // No memory for constants
                SigmaConstants.Count = 1;
            }

            // Random numbers for generation
            RandomUniform.Count = Input != null ? Input.Count : 1;

            // Random numbers for sampling
            RandomNormal.Count = Neurons;

            // Input after adding noise
            NoisyInput.Count = Neurons;
            Regularization.Count = 1;

            // Parameter allocations
            Weights.Count = 1;
            Bias.Count = 1;

            // SGD allocations
            Delta.Count = Neurons;
            PreviousWeightDelta.Count = Neurons; // momentum method
            PreviousBiasDelta.Count = Neurons; // momentum method

            // RMSProp allocations
            MeanSquareWeight.Count = Weights.Count;
            MeanSquareBias.Count = Bias.Count;

            // Adadelta allocation
            // AdadeltaWeight.Count = Weights.Count;
            // AdadeltaBias.Count = Bias.Count;

            // Priors for generation
            PriorGaussHiddenStatesMin.Count = Input != null ? Input.Count : 1;
            PriorGaussHiddenStatesMax.Count = Input != null ? Input.Count : 1;
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(Input.Count % 2 == 0, this, "Number of neurons in previous layer has to be even (two params mu, sigma).");
        }

        // Tasks
        public MyGaussianInitTask InitTask { get; protected set; }

        public void CreateTasks()
        {
            ForwardTask = new MyGaussianForwardTask();
            DeltaBackTask = new MyGaussianBackDeltaTask();
            InitTask = new MyGaussianInitTask();
        }

        // Parameterless constructor
        public MyGaussianHiddenLayer() { }

        // description
        public override string Description
        {
            get
            {
                return "Gaussian layer";
            }
        }
    }
}
