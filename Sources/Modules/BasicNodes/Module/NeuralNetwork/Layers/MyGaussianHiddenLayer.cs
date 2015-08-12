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
        [MyPersistable]
        public MyGenerateSignal Generate { get; private set; }
        public class MyGenerateSignal : MySignal { }

        public MyMemoryBlock<float> RandomNormal { get; private set; }
        public MyMemoryBlock<float> Regularization { get; private set; }

        public override ConnectionType Connection
        {
            get { return ConnectionType.GAUSSIAN; }
        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            if (Neurons > 0)
            {
                if (Input != null)
                {
                    // two parameters (mu, sigma) from previous layer for each neuron
                    Neurons = Input.Count / 2;

                    // Random numbers for sampling
                    RandomNormal.Count = Neurons;
                    Regularization.Count = 1;

                    // parameter allocations
                    Weights.Count = 1;
                    Bias.Count = 1;

                    // SGD allocations
                    Delta.Count = Neurons;
                    PreviousWeightDelta.Count = Neurons; // momentum method
                    PreviousBiasDelta.Count = Neurons; // momentum method

                    // RMSProp allocations
                    MeanSquareWeight.Count = Weights.Count;
                    MeanSquareBias.Count = Bias.Count;
                }
            }
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(Input.Count % 2 == 0, this, "Number of neurons in previous layer has to be even (two params mu, sigma).");
        }

        // Tasks
        public MyFCUpdateWeightsTask UpdateWeights { get; protected set; }
        public virtual void CreateTasks()
        {
            ForwardTask = new MyGaussianForwardTask();
            DeltaBackTask = new MyGaussianBackDeltaTask();
        }

        // Parameterless constructor
        public MyGaussianHiddenLayer() { }

        // description
        public override string Description
        {
            get
            {
                return "Gaussian hidden layer";
            }
        }
    }
}
