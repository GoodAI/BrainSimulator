using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using YAXLib;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;

using GoodAI.Core;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Core.Memory;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.Matrix;
using GoodAI.Core.Execution;

namespace GoodAI.Modules.GeneticWorld
{

    // Abstract class for supporting GeneticTrainingWorld Tasks
    public abstract class GeneticWorldTask : MyTask<MyGeneticTrainingWorld>
    {

        protected List<MyAbstractWeightLayer> layerList;
        // Each population member is a weight matrix 
        protected List<MyMemoryBlock<float>> population;
        protected MyMemoryBlock<float> fitnesses;
        protected MyExecutionPlan m_executionPlan;

        // Extracts the weights and biases from the layers in a network and places them into a pre-sized
        // memory block. Doesn't extract reccurent or other special weights.
        protected void getFFWeights(MyMemoryBlock<float> weights)
        {
            int weightArrayPointer = 0;
            foreach (MyAbstractWeightLayer m in layerList)
            {
                m.Weights.SafeCopyToHost();
                m.Bias.SafeCopyToHost();

                for (int i = 0; i < m.Weights.Count; i++)
                {
                    weights.Host[weightArrayPointer++] = m.Weights.Host[i];
                }

                for (int i = 0; i < m.Bias.Count; i++)
                {
                    weights.Host[weightArrayPointer++] = m.Bias.Host[i];
                }
            }
        }


        // Writes the weights and biases from an array and places them into a the layers of the
        // network. Doesn't write reccurent or other special weights, pair with extractWeights
        // to ensure correctness
        protected void setFFWeights(MyMemoryBlock<float> weights)
        {
            int weightArrayPointer = 0;
            foreach (MyAbstractWeightLayer m in layerList)
            {
                for (int i = 0; i < m.Weights.Count; i++)
                {
                    m.Weights.Host[i] = weights.Host[weightArrayPointer++];
                }

                for (int i = 0; i < m.Bias.Count; i++)
                {
                    m.Bias.Host[i] = weights.Host[weightArrayPointer++];
                }

                m.Weights.SafeCopyToDevice();
                m.Bias.SafeCopyToDevice();
            }
        }


        // Determines the fitness of a population of candidates
        protected void determineFitnesses()
        {
            // Test for fitness...       
            for (int pop = 0; pop < Owner.PopulationSize; pop++)
            {
                fitnesses.Host[pop] = testFitness(pop);
            }
        }


        // Tests the fitness of a set of weights for a network. 
        protected float testFitness(int member)
        {
            // Set weights
            setFFWeights(population[member]);

            while (true) // Concise, but dangerous!
            {
                // Execute all nodes in the simulation, but not this world
                m_executionPlan.StandardStepPlan.Children[1].Execute();

                // Check for reset signal in the world
                Owner.SwitchMember.SafeCopyToHost();
                if (Owner.SwitchMember.Host[0] != 0)
                {
                    break;
                }
            }
            Owner.Fitness.SafeCopyToHost();
            return Owner.Fitness.Host[0];
        }
    }

}
