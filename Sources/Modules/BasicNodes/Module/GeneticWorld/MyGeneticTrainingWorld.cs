using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Core.Execution;
using ManagedCuda.CudaBlas;
using GoodAI.Modules.Matrix;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Group;
using YAXLib;
using System.ComponentModel;

namespace GoodAI.Modules.GeneticWorld
{
    /// <author>GoodAI</author>
    /// <meta>JD</meta>
    ///<status>Working</status>
    /// <summary><b>Genetic Training World</b>
    ///
    /// <p>
    /// The genetic training world overrides the normal backpropagation training of a neural network group.
    /// Each timestep, the world executes it's current genetic task. A user defined fitness function evaluates
    /// the performance of each population member which give each member the score of Fitness when SwitchMember != 0 
    /// </p> 
    ///
    /// <p>
    /// The fitness function is defined by the user, usually as a C# node. When a member has a score = TargetFitness, the
    /// genetic task will stop with the weights of that member written to the network. The world that the network is trained against
    /// needs to be converted into a node by copying the KnownWorlds group up into the nodes group in the conf.xml file.
    /// </p>
    /// 
    /// <p><b>I/O:</b>
    /// <ul>
    /// <li><b>SwitchMember</b> - If != 0, the genetic task will start testing the next member of the population, recording the current value of the Fitness output block</li>
    /// <li><b>Fitness</b> - The current fitness of the population member. Recorded when SwitchMember is != 0</li>
    /// </ul>
    /// </p>
    /// 
    /// <p><b>Parmaters:</b>
    /// <ul>
    /// <li><b>TrainGenetically</b> - If FALSE, disables all genetic training methods.</li>
    /// <li><b>PopulationSize</b> - Size of the population to be evolved.</li>
    /// <li><b>TargetFitness</b> - Target fitness of the populations</li>
    /// <li><b>Generations</b> - Set a generation limit if the training method supports it. A set limit is > 0</li>
    /// <li><b>Survivors</b> - Number of survivng population members per generation</li>
    /// <li><b>MutationRate</b> - Rate of mutation during repopulation.</li>
    /// </ul>
    /// </p>
    /// </summary>
    public class MyGeneticTrainingWorld : MyWorld
    {

        [MyInputBlock(0)]
        public MyMemoryBlock<float> SwitchMember
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Fitness
        {
            get { return GetInput(1); }
        }

        [YAXSerializableField(DefaultValue = 10)]
        [MyBrowsable, Category("WorldParams"), Description("Size of the population to be evolved.")]
        public int PopulationSize { get; set; }

        [YAXSerializableField(DefaultValue = 100)]
        [MyBrowsable, Category("WorldParams"), Description("Target fitness of the populations")]
        public int TargetFitness { get; set; }

        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("WorldParams"), Description("Set a generation limit if the training method supports it. A set limit is > 0")]
        public int Generations { get; set; }

        [YAXSerializableField(DefaultValue = 5)]
        [MyBrowsable, Category("WorldParams"), Description("Number of surviving candidates each round.")]
        public int Survivors { get; set; }

        [YAXSerializableField(DefaultValue = 0.2f)]
        [MyBrowsable, Category("WorldParams"), Description("Rate of mutation during repopulation.")]
        public float MutationRate { get; set; }


        // Only one task just now, but could be extended like the MyNeuralNetworkGroup tasks
        [MyTaskGroup("Training")]
        public Cosyne CosyneTask { get; protected set; }

        public override void UpdateMemoryBlocks()
        {
            //throw new NotImplementedException();
        }

        public override void Validate(MyValidator validator)
        {
            validator.AssertError(Fitness != null, this, "Requires fitness input.");
            validator.AssertError(SwitchMember != null, this, "There must be a signal to change tested members. Otherwise infinite loops will occur.");
        }

        /// <author>GoodAI</author>
        /// <meta>JD</meta>
        ///<status>Working</status>
        /// <summary><b>CoSyNE Genetic Training</b>
        ///
        ///<p> CoSyNE Training is genetic method for continuous environments which convets the weight matrix of the network using a Discrete
        /// Cosine Transform (DCT) to a number of significant coefficents, the number of which are defined by the user.
        /// The rest of the population is filled with chromosomes between <b>Alpha</b> and <b>-Alpha</b> and each
        /// member converts the DCT back, tests the fitness of the candidate, and if the fitness is >= <b>TargetFitness</b>, the satisfying network is returned. Otherwise
        /// the CoSyNE recombination, mutation, and permutation methods are employed to create a new population and the
        /// process starts over.
        /// </p>
        /// 
        /// <p>
        /// Each timestep, a single generation of the CoSyNE method is executed.
        /// Non reccurent feedforward networks are all that is currently supported. The world will only train the first NN group that it finds at the top
        /// layer of the overall Network.
        /// </p>
        /// 
        /// <p><b>Parmaters:</b>
        /// <ul>
        /// <li><b>CoefficientsSaved</b> - The number of coefficeints to be evolved. If equal to the number of weights in the network, the weights are evolved directly without the DCT.</li>
        /// <li><b>Alpha</b> - Range of the generated coefficients (if evolving coefficients)</li>
        /// <li><b>WeightMagnitude</b> - Range of the generated weights (if evolving weights directly)</li>
        /// <li><b>DirectEvolution</b> - Evolve weights without DCT translation? Automatically TRUE if CoefficientsSaved >= number of weights in the NN</li>
        /// </ul>
        /// </p>
        /// </summary>
        [Description("CoSyNE Training"), MyTaskInfo(OneShot = false)]
        public class Cosyne : GeneticWorldTask
        {
            [YAXSerializableField(DefaultValue = 3)]
            [MyBrowsable, Category("GeneticVariables"), Description("The number of coefficeints to be evolved. If equal to the number of weights in the network, the weights are evolved directly without the DCT.")]
            public int CoefficientsSaved { get; set; }

            [YAXSerializableField(DefaultValue = 20)]
            [MyBrowsable, Category("GeneticVariables"), Description("Range of generated coefficient magnitudes.")]
            public int Alpha { get; set; }

            [YAXSerializableField(DefaultValue = 5)]
            [MyBrowsable, Category("GeneticVariables"), Description("Range of generated network weight magnitudes.")]
            public int WeightMagnitude { get; set; }

            [YAXSerializableField(DefaultValue = false)]
            [MyBrowsable, Category("GeneticVariables"), Description("Evolve weights without DCT translation? Automatically TRUE if CoefficientsSaved >= number of weights in the NN")]
            public bool DirectEvolution { get; set; }

            private MyNeuralNetworkGroup nn = null;
            private MyMemoryBlock<float> chromosomePop = null;
            private MyMemoryBlock<float> noise = null;
            private MyMemoryBlock<float> multiplier = null;
            private List<MyMemoryBlock<float>> outputPop = null;
            private MyMemoryBlock<float> cudaMatrices = null;
            private MyMemoryBlock<float> tempMB = null;
            private MyMemoryBlock<float> tempPop = null;
            private MyMemoryBlock<int> marking = null;

            private int currentGen;
            private Random m_rand;
            private int m_weights, arr_size;
            private MyCudaKernel m_geneticKernel, m_extractKernel, m_cosineGenKernel, m_coeffGenKernel, m_implantKernel;


            // Sets up the genetic task
            public override void Init(int nGPU)
            {
                currentGen = 0;
                m_weights = 0;


                // Load the relevant kernels
                m_coeffGenKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Genetic\CosyneGenetics", "generateCoefficients");
                m_geneticKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Genetic\CosyneGenetics", "grow");
                m_extractKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Genetic\CosyneGenetics", "extractCoeffs");
                m_cosineGenKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Genetic\CosyneGenetics", "createCosineMatrix");
                m_implantKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Genetic\CosyneGenetics", "implantCoeffs");

                // Init the random generator
                m_rand = new Random();

                // Set up coefficient Generation
                m_coeffGenKernel.SetupExecution(Owner.PopulationSize);
                // Set up genetic recombination
                m_geneticKernel.SetupExecution(Owner.PopulationSize);


                // This finds the first nn group in the network. Possibility of getting a list of networks and evolving them all seperately?      
                List<MyNode> ch = Owner.Owner.Network.Children;
                foreach (MyNode n in ch)
                {
                    if (n is MyNeuralNetworkGroup)
                    {
                        nn = n as MyNeuralNetworkGroup;
                        MyLog.INFO.WriteLine("Evolving the layers of node: " + nn.Name);
                        break;
                    }
                }
                if (nn == null)
                {
                    throw new NullReferenceException("There is no top level NeuralNetworkGroup.");
                }

                // Construct the layerlist which is to be read from and written to
                constructLayerList(nn);

                // This is how big the weight matrix will be
                arr_size = (int)Math.Ceiling(Math.Sqrt(m_weights));

                // Get the relevant execution plan
                m_executionPlan = Owner.Owner.SimulationHandler.Simulation.ExecutionPlan;

                #region MemoryBlocks
                // Initialise the population
                population = new List<MyMemoryBlock<float>>();
                outputPop = new List<MyMemoryBlock<float>>();
                for (int i = 0; i < Owner.PopulationSize; i++)
                {
                    population.Add(new MyMemoryBlock<float>());
                    population[i].Owner = Owner;
                    population[i].Count = arr_size * arr_size;
                    population[i].AllocateMemory();

                    outputPop.Add(new MyMemoryBlock<float>());
                    outputPop[i].Owner = Owner;
                    outputPop[i].Count = arr_size * arr_size;
                    outputPop[i].AllocateMemory();
                }

                // Allocate space to manipulate weight matrices on the device
                cudaMatrices = new MyMemoryBlock<float>();
                cudaMatrices.Owner = Owner;
                cudaMatrices.Count = arr_size * arr_size * Owner.PopulationSize;
                cudaMatrices.AllocateDevice();

                // Allocate a memory block for the Cosine matrix
                multiplier = new MyMemoryBlock<float>();
                multiplier.Owner = Owner;
                multiplier.Count = arr_size * arr_size;
                multiplier.AllocateDevice();

                // Fill the cosine Matrices
                m_cosineGenKernel.SetupExecution(arr_size);
                m_cosineGenKernel.Run(multiplier, arr_size);

                // Allocate space needed for chromosomes
                chromosomePop = new MyMemoryBlock<float>();
                chromosomePop.Owner = Owner;
                if (DirectEvolution)
                    chromosomePop.Count = m_weights * Owner.PopulationSize;
                else
                    chromosomePop.Count = CoefficientsSaved * Owner.PopulationSize;
                chromosomePop.AllocateMemory();

                // Allocate some space for noise to seed the cuda_rand generator
                noise = new MyMemoryBlock<float>();
                noise.Owner = Owner;
                noise.Count = Owner.PopulationSize;
                noise.AllocateMemory();

                // Write some noise to the initial array
                for (int i = 0; i < Owner.PopulationSize; i++)
                {
                    noise.Host[i] = (float)m_rand.NextDouble() * 100000 + (float)m_rand.NextDouble() * 40;
                }
                noise.SafeCopyToDevice();

                // Allocate space for the fitnesses
                fitnesses = new MyMemoryBlock<float>();
                fitnesses.Owner = Owner;
                fitnesses.Count = Owner.PopulationSize;
                fitnesses.AllocateMemory();


                // Allocate some temporary storage
                tempMB = new MyMemoryBlock<float>();
                tempPop = new MyMemoryBlock<float>();
                tempMB.Owner = Owner;
                tempMB.Count = CoefficientsSaved;
                tempMB.AllocateDevice();

                tempPop.Owner = Owner;
                tempPop.Count = arr_size * arr_size;
                tempPop.AllocateDevice();

                marking = new MyMemoryBlock<int>();
                marking.Owner = Owner;
                marking.Count = CoefficientsSaved * Owner.PopulationSize;
                marking.AllocateDevice();
                #endregion

                // Check saved Coeffs size
                if (CoefficientsSaved > m_weights)
                {
                    MyLog.WARNING.Write("Saving more Coefficients than exist in the weight matrix. Setting to max permissable value\n");
                    CoefficientsSaved = m_weights;
                }

                if (CoefficientsSaved == m_weights)
                {
                    MyLog.INFO.Write("Saving a coefficient for every weight. Evolving weights directly\n");
                    DirectEvolution = true;
                }

                if (DirectEvolution)
                    CoefficientsSaved = m_weights;


                // Generate the rest of the population
                if (DirectEvolution)
                    m_coeffGenKernel.Run(chromosomePop, CoefficientsSaved, noise, Owner.PopulationSize, WeightMagnitude);
                else
                    m_coeffGenKernel.Run(chromosomePop, CoefficientsSaved, noise, Owner.PopulationSize, Alpha);


                //Disable Backprop tasks in Network
                if (nn.GetActiveBackpropTask() != null)
                {
                    if (!nn.GetActiveBackpropTask().DisableLearning)
                    {
                        MyLog.WARNING.WriteLine("Disabling backprop learning for Neural Network");
                        nn.GetActiveBackpropTask().DisableLearning = true;
                    }
                }
            }


            public override void Execute()
            {
                currentGen++;
                // If not genetically training. Return

                //Get first population member from the network
                getFFWeights(population[0]);
                population[0].SafeCopyToDevice();
                if (!DirectEvolution)
                {               
                    MyCublasFactory.Instance.Gemm(Operation.NonTranspose, Operation.NonTranspose,
                              arr_size, arr_size, arr_size, 1.0f,
                              multiplier.GetDevice(Owner), arr_size,
                              population[0].GetDevice(Owner), arr_size,
                              0.0f, outputPop[0].GetDevice(Owner), arr_size
                              );

                    MyCublasFactory.Instance.Gemm(Operation.NonTranspose, Operation.Transpose,
                              arr_size, arr_size, arr_size, 1.0f,
                              outputPop[0].GetDevice(Owner), arr_size,
                              multiplier.GetDevice(Owner), arr_size,
                              0.0f, population[0].GetDevice(Owner), arr_size
                              );
                }
                //Read the saved coeffs from the initial weight matrix into the first chromosome
                population[0].CopyToMemoryBlock(cudaMatrices, 0, 0, arr_size * arr_size);
                m_extractKernel.SetupExecution(1);
                m_extractKernel.Run(cudaMatrices, chromosomePop, CoefficientsSaved, arr_size);


                // Recombine and grow the population
                if (DirectEvolution)
                    m_geneticKernel.Run(cudaMatrices, arr_size, m_weights, Owner.PopulationSize, chromosomePop, noise, Owner.MutationRate, Owner.Survivors, fitnesses, marking, WeightMagnitude);
                else
                    m_geneticKernel.Run(cudaMatrices, arr_size, CoefficientsSaved, Owner.PopulationSize, chromosomePop, noise, Owner.MutationRate, Owner.Survivors, fitnesses, marking, Alpha);


                chromosomePop.SafeCopyToHost();
                cudaMatrices.Fill(0.0f);
                m_implantKernel.SetupExecution(Owner.PopulationSize);
                m_implantKernel.Run(cudaMatrices, chromosomePop, CoefficientsSaved, arr_size);


                for (int i = 0; i < Owner.PopulationSize; i++)
                {
                    // Read the cudaMatrices into the population
                    population[i].CopyFromMemoryBlock(cudaMatrices, i * arr_size * arr_size, 0, arr_size * arr_size);

                    if (!DirectEvolution)
                    {
                        MyCublasFactory.Instance.Gemm(Operation.Transpose, Operation.NonTranspose,
                        arr_size, arr_size, arr_size, 1.0f,
                        multiplier.GetDevice(Owner), arr_size,
                        population[i].GetDevice(0), arr_size,
                        0.0f, outputPop[i].GetDevice(0), arr_size
                        );

                        MyCublasFactory.Instance.Gemm(Operation.NonTranspose, Operation.NonTranspose,
                          arr_size, arr_size, arr_size, 1.0f,
                          outputPop[i].GetDevice(0), arr_size,
                          multiplier.GetDevice(Owner), arr_size,
                          0.0f, population[i].GetDevice(0), arr_size
                          );
                    }
                    population[i].SafeCopyToHost();
                    noise.Host[i] = (float)m_rand.NextDouble();
                }
                noise.SafeCopyToDevice();



                // Determine the fitness of each member
                determineFitnesses();
                chromosomePop.SafeCopyToHost();

                #region Sort Chromosomes
                //sort the chromosomes and populations by fitness 
                //bubble sort, can be improved
                float tmpfit;
                int len = Owner.PopulationSize;
                int newlen;

                while (len != 0)
                {
                    newlen = 0;
                    for (int i = 1; i < len; i++)
                    {
                        if (fitnesses.Host[i - 1] < fitnesses.Host[i])
                        {
                            // Swap fitnesses on the host
                            tmpfit = fitnesses.Host[i - 1];
                            fitnesses.Host[i - 1] = fitnesses.Host[i];
                            fitnesses.Host[i] = tmpfit;
                            newlen = i;
                            // Swap Chromosomes on the device
                            for (int x = 0; x < CoefficientsSaved; x++)
                            {
                                tmpfit = chromosomePop.Host[i * CoefficientsSaved + x];
                                chromosomePop.Host[i * CoefficientsSaved + x] = chromosomePop.Host[(i - 1) * CoefficientsSaved + x];
                                chromosomePop.Host[(i - 1) * CoefficientsSaved + x] = tmpfit;
                            }

                            for (int x = 0; x < arr_size * arr_size; x++)
                            {
                                tmpfit = population[i - 1].Host[x];
                                population[i - 1].Host[x] = population[i].Host[x];
                                population[i].Host[x] = tmpfit;
                            }
                        }
                    }
                    len = newlen;
                }

                MyLog.INFO.WriteLine("Top {0} networks:", Math.Max(Owner.Survivors, Owner.PopulationSize / 10));
                for (int i = 0; i < Math.Max(Owner.Survivors, Owner.PopulationSize / 10); i++)
                {
                    MyLog.INFO.Write("Fitness of network {0} is: {1}", i, fitnesses.Host[i]);
                    if (i < Owner.Survivors)
                    {
                        MyLog.INFO.Write(" - surviving");
                    }
                    MyLog.INFO.Write(" \n");
                }

                #endregion

                // Best candidate to write to the network is the top of the population list
                MyLog.INFO.WriteLine("Fitness of selected network is: " + fitnesses.Host[0]);
                if (fitnesses.Host[0] >= Owner.TargetFitness)
                {
                    MyLog.INFO.WriteLine("Found satisfying network, halting...");
                    Owner.Owner.SimulationHandler.PauseSimulation();
                }

                setFFWeights(population[0]);
                MyLog.INFO.WriteLine("Written weights to network");
                if (currentGen >= Owner.Generations && Owner.Generations > 0)
                {
                    MyLog.INFO.WriteLine("Generation limit reached, halting...");
                    Owner.Owner.SimulationHandler.PauseSimulation();
                }
            }

            // Constructs topographical list of layers in a NN group and records the
            // FF weights
            private void constructLayerList(MyNeuralNetworkGroup group)
            {
                MyAbstractWeightLayer t;
                layerList = new List<MyAbstractWeightLayer>();
                List<MyNode> n = group.Children;
                foreach (MyNode l in n)
                {
                    if (l is MyAbstractWeightLayer)
                    {
                        t = l as MyAbstractWeightLayer;
                        m_weights += t.Weights.Count;
                        m_weights += t.Bias.Count;
                        layerList.Add(t);
                        MyLog.DEBUG.Write("Layer " + l.Name + " added\n");

                    }
                    else
                        MyLog.WARNING.Write("Layer " + l.Name + " is not a weighted layer\n");
                }
            }

        }

    }



}
