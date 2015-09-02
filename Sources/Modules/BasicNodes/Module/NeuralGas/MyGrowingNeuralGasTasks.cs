using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Modules.NeuralGas
{
    /// <summary>
    /// Initialization of the growing neural gas node
    /// Parameters:
    /// <ul>
    /// <li>MIN_INIT_INPUT: minimal value of the input and reference vector cell</li>
    /// <li>MAX_INIT_INPUT: maximal value of the input and reference vector cell</li>
    /// </ul>
    /// </summary>
    [Description("Init growing neural gas node"), MyTaskInfo(OneShot = true)]
    public class MyInitGrowingNeuralGasNodeTask : MyTask<MyGrowingNeuralGasNode>
    {
        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float MIN_INIT_INPUT { get; set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1.00f), YAXElementFor("Structure")]
        public float MAX_INIT_INPUT { get; set; }

        public override void Init(int nGPU)
        {
            
        }

        public override void Execute()
        {
            Random randomNumber = new Random();
            for (int i = 0; i < Owner.ReferenceVector.Count; i++)
            {
                Owner.ReferenceVector.Host[i] = MIN_INIT_INPUT + (MAX_INIT_INPUT - MIN_INIT_INPUT)*(float)randomNumber.NextDouble();
            }
            Owner.ReferenceVector.SafeCopyToDevice();


            // make connections among all the living cells in the beginning
            for (int i = 0; i < Owner.INIT_LIVE_CELLS; i++)
            {
                for (int c = 0; c < Owner.INIT_LIVE_CELLS; c++)
                {
                    if (i != c)
                    {
                        Owner.ConnectionMatrix.Host[i * Owner.MAX_CELLS + c] = 1;
                        //Owner.ConnectionMatrix.Host[c * Owner.MAX_CELLS + 1] = 1;    
                    }
                    
                }
            }
            
            
            Owner.ConnectionMatrix.SafeCopyToDevice();

            Owner.ConnectionAge.Fill(0);
            Owner.Distance.Fill(0);

            Array.Clear(Owner.ActivityFlag.Host, 0, Owner.ActivityFlag.Count);
            for (int i = 0; i < Owner.INIT_LIVE_CELLS; i++)
            {
                Owner.ActivityFlag.Host[i] = 1;
            }
            Owner.ActivityFlag.SafeCopyToDevice();
            Owner.activeCells = Owner.INIT_LIVE_CELLS;

            Owner.LocalError.Fill(0);

            Owner.Difference.Fill(0);

            Owner.TwoNodesDifference.Fill(0);
            Owner.TwoNodesDistance.Fill(0);

            Owner.WinningFraction.Fill(0);
            Owner.BiasTerm.Fill(0);
            Owner.BiasedDistance.Fill(0);

            Owner.OutputOne.Fill(0);
            Owner.WinnerOne.Fill(0);
            Owner.OutputTwo.Fill(0);
            Owner.WinnerTwo.Fill(0);

            Owner.NeuronAge.Fill(0);
            Owner.WinningCount.Fill(0);

            Owner.DimensionWeight.Fill(1.00f);

            Owner.Utility.Fill(0);

            //Owner.DimensionWeight.Host[0] = 1.00f;
            //Owner.DimensionWeight.Host[1] = 1.00f;
            //Owner.DimensionWeight.Host[2] = 2.00f;
            //Owner.DimensionWeight.SafeCopyToDevice();
             
            
        }
    }

    /// <summary>
    /// Find the neural gas cell with best matching reference vector to the input
    /// </summary>
    [Description("Find winners task"), MyTaskInfo(OneShot = false)]
    public class MyFindWinnersTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_differenceKernel;
        private MyCudaKernel m_distanceKernel;
        
        public override void Init(int nGPU)
        {
            m_differenceKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel","VectorInputDiffKernel");
            m_distanceKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "ComputeDistanceKernel");
        }

        public override void Execute()
        {
            // compute difference between input and reference vectors
            m_differenceKernel.SetupExecution(Owner.MAX_CELLS * Owner.Input.Count
                );

            m_differenceKernel.Run(Owner.Input,
                Owner.Input.Count,
                Owner.ReferenceVector,
                Owner.MAX_CELLS,
                Owner.Difference
                );

            // compute cumulative error of the difference between input and reference vector
            m_distanceKernel.SetupExecution(Owner.MAX_CELLS
                );

            m_distanceKernel.Run(Owner.Input.Count,
                Owner.Distance,
                Owner.DimensionWeight,
                Owner.MAX_CELLS,
                Owner.Difference
                );

            
            Owner.ActivityFlag.SafeCopyToHost();
            Owner.Distance.SafeCopyToHost();

            float min = float.MaxValue;
            float secondMin = float.MaxValue;

            for (int i = 0; i < Owner.MAX_CELLS; i++)
            {
                if (Owner.ActivityFlag.Host[i] == 1)
                {
                    if (Owner.Distance.Host[i] <= min)
                    {
                        Owner.s2 = Owner.s1;
                        Owner.s1 = i;
                        secondMin = min;
                        min = Owner.Distance.Host[i];
                    }
                    else if(Owner.Distance.Host[i] <= secondMin)
                    {
                        Owner.s2 = i;
                        secondMin = Owner.Distance.Host[i];
                    }
                }
                    
            }
        }
    }

    /// <summary>
    /// For GNG with consciousness modification it is needed to store the winning fraction of each neuron cell.<br />
    /// B_PARAM: learning rate of the winning fraction counter (the counter is based on the Hebbian learning rule)
    /// </summary>
    [Description("Adapt winning fraction and count"), MyTaskInfo(OneShot = false)]
    public class MyAdaptWinningFractionTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_kernel;

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.0001f), YAXElementFor("Structure")]
        public float B_PARAM { get; set; }

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "AdaptWinningFractionKernel");
        }
        
        public override void Execute()
        {
            m_kernel.SetupExecution(Owner.MAX_CELLS
                );

            m_kernel.Run(Owner.s1,
                Owner.WinningFraction,
                Owner.WinningCount,
                B_PARAM,
                Owner.MAX_CELLS
                );
        }

    }

    /// <summary>
    /// For GNG with consciousness modification this task compute biased winning neuron cell
    /// C_FACTOR: the strength of consciousness, if set to zero the algorithm is standard GNG
    /// </summary>
    [Description("Compute bias term"), MyTaskInfo(Disabled = true, OneShot = false)]
    public class MyBiasTermTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_kernel;
        
        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float C_FACTOR { get; set; }

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "ComputeBiasTermKernel");
        }

        public override void Execute()
        {
            m_kernel.SetupExecution(Owner.MAX_CELLS
                );

            m_kernel.Run(Owner.BiasTerm,
                C_FACTOR,
                Owner.WinningFraction,
                Owner.activeCells,
                Owner.MAX_CELLS
                );
        }
    }

    /// <summary>
    /// Find the winner with considering the consciousness bias
    /// </summary>
    [Description("Find concsious winners"), MyTaskInfo(Disabled = true, OneShot = false)]
    public class MyFindConsciousWinnersTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_kernel;

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "ComputeBiasedDistanceKernel");
        }

        public override void Execute()
        {
            m_kernel.SetupExecution(Owner.MAX_CELLS
                );

            m_kernel.Run(Owner.Distance,
                Owner.BiasedDistance,
                Owner.BiasTerm,
                Owner.MAX_CELLS
                );

            Owner.ActivityFlag.SafeCopyToHost();
            Owner.BiasedDistance.SafeCopyToHost();

            float min = float.MaxValue;
            float secondMin = float.MaxValue;

            for (int i = 0; i < Owner.MAX_CELLS; i++)
            {
                if (Owner.ActivityFlag.Host[i] == 1)
                {
                    if (Owner.BiasedDistance.Host[i] <= min)
                    {
                        Owner.s2 = Owner.s1;
                        Owner.s1 = i;
                        secondMin = min;
                        min = Owner.BiasedDistance.Host[i];
                    }
                    else if (Owner.BiasedDistance.Host[i] <= secondMin)
                    {
                        Owner.s2 = i;
                        secondMin = Owner.BiasedDistance.Host[i];
                    }
                }

            }
        }
    }

    /// <summary>
    /// Send all the neccessary data to the corresponding output fields
    /// </summary>
    [Description("Send data to output"), MyTaskInfo(OneShot = false)]
    public class MySendDataToOutputTask : MyTask<MyGrowingNeuralGasNode>
    {
        public override void Init(int nGPU)
        { 
            
        }
        
        public override void Execute()
        {
            Owner.ReferenceVector.CopyToMemoryBlock(Owner.OutputOne, Owner.s1 * Owner.INPUT_SIZE, 0, Owner.INPUT_SIZE);
            Owner.WinnerOne.Fill(0);
            Owner.WinnerOne.GetDevice(Owner).CopyToDevice(1.00f, Owner.s1 * sizeof(float));
            Owner.ReferenceVector.CopyToMemoryBlock(Owner.OutputTwo, Owner.s2 * Owner.INPUT_SIZE, 0, Owner.INPUT_SIZE);
            Owner.WinnerTwo.Fill(0);
            Owner.WinnerTwo.GetDevice(Owner).CopyToDevice(1.00f, Owner.s2 * sizeof(float));
        }
    }

    /// <summary>
    /// For the winning neural cell cummulate its error and the utility value
    /// </summary>
    [Description("Add local error and utility to the winner"), MyTaskInfo(OneShot = false)]
    public class MyAddLocalErrorAndUtilityTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_addLocalErrorKernel;
        private MyCudaKernel m_addUtilityKernel;

        public override void Init(int nGPU)
        {
            m_addLocalErrorKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "AddLocalErrorKernel");
            m_addUtilityKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "AddUtilityKernel");
        }

        public override void Execute()
        {
            m_addLocalErrorKernel.SetupExecution(1
                );

            m_addLocalErrorKernel.Run(Owner.s1,
                Owner.Distance,
                Owner.LocalError
                );

            m_addUtilityKernel.SetupExecution(1
                );

            m_addUtilityKernel.Run(Owner.s1,
                Owner.s2,
                Owner.Distance,
                Owner.Utility
                );
            
        }
    }

    /// <summary>
    /// Create connection between the winner 1 and 2 (if there isn't one already)
    /// </summary>
    [Description("Create connection and refresh"), MyTaskInfo(OneShot = false)]
    public class MyCreateConnecionTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_kernel;

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "CreateAndRefreshConnectionKernel");
        }

        public override void Execute()
        { 
            // TODO : FIND ANOTHER WAY HOW TO DO IT THAN THROUGH KERNEL
            m_kernel.SetupExecution(1
                );

            m_kernel.Run(Owner.s1,
                Owner.s2,
                Owner.ConnectionMatrix,
                Owner.ConnectionAge,
                Owner.MAX_CELLS
                );
        }
    }

    /// <summary>
    /// Adapt the reference vector of the winner and its connected neural cell<br/>
    /// The adaptation can be set in a way that not trained cells (distinquished by the low winning count) are adapting faster than trained ones.
    /// Parameters:
    /// <ul>
    /// <li>E_b_YOUNG: the learning rate for the untrained winner cell</li>
    /// <li>E_b_OLD: the learning rate for the trained winner cell</li>
    /// <li>E_n_YOUNG: the learning rate for the untrained neighboring cell</li>
    /// <li>E_n_OLD: the learning rate for the trained neighboring cell</li>
    /// <li>DECAY_FACTOR: the decay factor of the learning rate, the learning rate changes exponentialy from _YOUNG to _OLD value</li>
    /// </ul>
    /// </summary>
    [Description("Adapt the reference vector"), MyTaskInfo(OneShot = false)]
    public class MyAdaptRefVectorTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_kernel;

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.05f), YAXElementFor("Structure")]
        public float E_b_YOUNG { get; set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.000001f), YAXElementFor("Structure")]
        public float E_b_OLD { get; set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.0006f), YAXElementFor("Structure")]
        public float E_n_YOUNG { get; set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.000001f), YAXElementFor("Structure")]
        public float E_n_OLD { get; set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.001f), YAXElementFor("Structure")]
        public float DECAY_FACTOR { get; set; }

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "AdaptRefVectorKernel");
        }

        public override void Execute()
        {
            m_kernel.SetupExecution(Owner.INPUT_SIZE
                );

            m_kernel.Run(Owner.s1,
                Owner.ReferenceVector,
                E_b_OLD,
                E_b_YOUNG,
                DECAY_FACTOR,
                Owner.WinningCount,
                Owner.Difference,
                Owner.INPUT_SIZE
                );

            Owner.ConnectionMatrix.SafeCopyToHost();
            Owner.ActivityFlag.SafeCopyToHost();
            for (int i = 0; i < Owner.MAX_CELLS; i++)
            {
                if (Owner.ConnectionMatrix.Host[Owner.s1 * Owner.MAX_CELLS + i] == 1 && Owner.ActivityFlag.Host[i] == 1)
                {
                    m_kernel.Run(i,
                        Owner.ReferenceVector,
                        E_n_OLD,
                        E_n_YOUNG,
                        DECAY_FACTOR,
                        Owner.WinningCount,
                        Owner.Difference,
                        Owner.INPUT_SIZE
                        );
                }
            }
        }
    }

    /// <summary>
    /// Increment the connection age emanating from the winner neuron cell
    /// </summary>
    [Description("Increment connection age"), MyTaskInfo(OneShot = false)]
    public class MyIncrementConnectionAgeTask : MyTask<MyGrowingNeuralGasNode>
    {
        private MyCudaKernel m_kernel;

        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "IncrementConnectionAgeKernel");
        }

        public override void Execute()
        {
            m_kernel.SetupExecution(Owner.MAX_CELLS
                );

            m_kernel.Run(Owner.s1,
                Owner.ConnectionMatrix,
                Owner.ConnectionAge,
                Owner.MAX_CELLS
                );
        }
    }

    /// <summary>
    /// Remove the connections if they are old and cells if they don't have any connections or if they are not utilized<br/>
    /// Note: the neural cell removing according to utility factor was not really efficient and hard to set
    /// Parameters:<br/>
    /// <ul>
    /// <li>MAX_AGE: maximum possible age for the connection, if it is higher the connection is removed</li>
    /// <li>USE_UTILITY: if True - use the removing based on cell utility, if False - don't use it</li>
    /// <li>UTILITY: the utility threshold for removing the cells</li>
    /// </ul>
    /// </summary>
    [Description("Remove connections and cells"), MyTaskInfo(OneShot = false)]
    public class MyRemoveConnsAndCellsTask : MyTask<MyGrowingNeuralGasNode>
    {
        [MyBrowsable, Category("Parameter")]
        [YAXSerializableField(DefaultValue = 88), YAXElementFor("Structure")]
        public int MAX_AGE { get; set; }

        [MyBrowsable, Category("Utility")]
        [YAXSerializableField(DefaultValue = Option.False), YAXElementFor("Structure")]
        public Option USE_UTILITY { get; set; }

        [MyBrowsable, Category("Utility")]
        [YAXSerializableField(DefaultValue = 10000.00f), YAXElementFor("Structure")]
        public float UTILITY { get; set; }

        public enum Option
        { 
            True,
            False
        }

        private MyCudaKernel m_removeEdgesKernel;
        private MyCudaKernel m_removeNodeByUtilitKernel;

        public override void Init(int nGPU)
        {
            m_removeEdgesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "RemoveEdgesKernel");
            m_removeNodeByUtilitKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "RemoveNodeByUtilityKernel");
        }

        public override void Execute()
        {
            
            m_removeEdgesKernel.SetupExecution(Owner.MAX_CELLS
                );

            m_removeEdgesKernel.Run(Owner.ConnectionMatrix,
                Owner.ConnectionAge,
                MAX_AGE,
                Owner.ActivityFlag,
                Owner.WinningFraction,
                Owner.WinningCount,
                Owner.Utility,
                Owner.LocalError,
                Owner.NeuronAge,
                Owner.MAX_CELLS
                );

            Owner.ActivityFlag.SafeCopyToHost();
            Owner.activeCells = Owner.ActivityFlag.Host.Sum();
            
            if (USE_UTILITY == Option.True)
            {
                Owner.LocalError.SafeCopyToHost();
                float maxError = Owner.LocalError.Host.Max();

                m_removeNodeByUtilitKernel.SetupExecution(Owner.MAX_CELLS
                    );

                m_removeNodeByUtilitKernel.Run(Owner.ConnectionMatrix,
                    Owner.ConnectionAge,
                    Owner.ActivityFlag,
                    Owner.Utility,
                    UTILITY,
                    Owner.LocalError,
                    Owner.NeuronAge,
                    Owner.WinningFraction,
                    Owner.WinningCount,
                    maxError,
                    Owner.MAX_CELLS
                    );

                Owner.ActivityFlag.SafeCopyToHost();
                Owner.activeCells = Owner.ActivityFlag.Host.Sum();
            }

            
        }
    }


    /// <summary>
    /// Adding new nodes (neural cells) to the neural gas<br/>
    /// Rules:<br/>
    /// <ul>
    /// <li>INCREMENTAL: the new node is added each fixed number of steps in the place between node with the highest accumulated error and its neighbor with highest accumulated error</li>
    /// <li>VECTOR_OUTLIER: if the input is too far from the closest cell the new one is created with the reference vector same as the input</li>
    /// <li>EQUILIBRIUM: the new node is added every time when the average accumulated error reach a pre-defined value</li>
    /// </ul>
    /// Incremental:<br/>
    /// <ul>
    /// <li>LAMBDA - number of time steps when the new cell is added</li>
    /// <li>ALFA - fraction of the error which is subtracted from the neighbors of newly created neural cell</li>
    /// </ul>
    /// Vector outliers:<br/>
    /// <ul>
    /// <li>DISTANCE - the distance percentage between winner 1 and 2 which, if reached new neural cell is created on the place of input vector</li>
    /// </ul>
    /// Equilibrium:<br/>
    /// <ul>
    /// <li>AVG_E_TH - average error threshold, if reached new cell is created on the place as in Incremental adding rule</li>
    /// </ul>
    /// </summary>
    [Description("Add new node"), MyTaskInfo(OneShot = false)]
    public class MyAddNewNodeTask : MyTask<MyGrowingNeuralGasNode>
    {
        [MyBrowsable, Category("Rules")]
        [YAXSerializableField(DefaultValue = Turning.On), YAXElementFor("Structure")]
        public Turning INCREMENTAL { get; set; }

        [MyBrowsable, Category("Rules")]
        [YAXSerializableField(DefaultValue = Turning.Off), YAXElementFor("Structure")]
        public Turning VECTOR_OUTLIER { get; set; }

        [MyBrowsable, Category("Rules")]
        [YAXSerializableField(DefaultValue = Turning.Off), YAXElementFor("Structure")]
        public Turning EQUILIBRIUM { get; set; }

        /*
        [MyBrowsable, Category("Rules")]
        [YAXSerializableField(DefaultValue = Turning.Off), YAXElementFor("Structure")]
        public Turning SOFT { get; set; }
        */

        /*
        [MyBrowsable, Category("Rules")]
        [YAXSerializableField(DefaultValue = Turning.Off), YAXElementFor("Structure")]
        public Turning ERROR_OUTLIER { get; set; }

        [MyBrowsable, Category("Rules")]
        [YAXSerializableField(DefaultValue = Turning.Off), YAXElementFor("Structure")]
        public Turning NEIGHBORS_ERROR { get; set; }
        */

        public enum Turning
        { 
            On,
            Off
        }

        [MyBrowsable, Category("Incremental")]
        [YAXSerializableField(DefaultValue = 300), YAXElementFor("Structure")]
        public int LAMBDA { get; set; }

        [MyBrowsable, Category("Incremental")]
        [YAXSerializableField(DefaultValue = 0.5f), YAXElementFor("Structure")]
        public float ALFA { get; set; }

        [MyBrowsable, Category("Vector outliers")]
        [YAXSerializableField(DefaultValue = 1.00f), YAXElementFor("Structure")]
        public float DISTANCE { get; set; }

        [MyBrowsable, Category("Equilibrium")]
        [YAXSerializableField(DefaultValue = 1000f), YAXElementFor("Structure")]
        public float AVG_E_TH { get; set; }

        /*
        [MyBrowsable, Category("Soft")]
        [YAXSerializableField(DefaultValue = 10000f), YAXElementFor("Structure")]
        public float GLOBAL_E_TH { get; set; }

        [MyBrowsable, Category("Soft")]
        [YAXSerializableField(DefaultValue = 1.00f), YAXElementFor("Structure")]
        public float DIFF_TOLERANCE { get; set; }
        */
        /*
        [MyBrowsable, Category("Error outlier")]
        [YAXSerializableField(DefaultValue = 2.00f), YAXElementFor("Structure")]
        public float AVG_PRODUCT { get; set; }

        [MyBrowsable, Category("Neighbors error")]
        [YAXSerializableField(DefaultValue = 2.00f), YAXElementFor("Structure")]
        public float NEIGHBORS_E_AVG_PRODUCT { get; set; }
        */
        private MyCudaKernel m_InterpolateVectorKernel;
        private MyCudaKernel m_newNodeConnectionKernel;

        private MyCudaKernel m_twoNodesDifferenceKernel;
        private MyCudaKernel m_twoNodesDistanceKernel;
        private MyCudaKernel m_copyVectorKernel;
        private MyCudaKernel m_addAndRefreshConnectionKernel;

        //float lastLocalErrorSum;

        public override void Init(int nGPU)
        {
            m_InterpolateVectorKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "InterpolateVectorKernel");
            m_newNodeConnectionKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "NewNodeConnectionKernel");

            m_twoNodesDifferenceKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "TwoNodesDifferenceKernel");
            m_twoNodesDistanceKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "TwoNodesDistanceKernel");
            m_copyVectorKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "CopyVectorKernel");
            m_addAndRefreshConnectionKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "AddAndRefreshConnectionKernel");
        }

        public override void Execute()
        {
            /*
            if (SimulationStep == 0)
            {
                lastLocalErrorSum = 0.00f;
            }
            */

            // find out if there is a place for new cells
            Owner.ActivityFlag.SafeCopyToHost();
            Owner.activeCells = Owner.ActivityFlag.Host.Sum();

            if ((SimulationStep != 0))
            {
                /*
                if (SOFT == Turning.On )
                {
                    float localErrorSum = 0.00f;
                    Owner.LocalError.SafeCopyToHost();
                    for (int i = 0; i < Owner.MAX_CELLS; i++)
                    {
                        if (Owner.ActivityFlag.Host[i] == 1)
                        {
                            localErrorSum += Owner.LocalError.Host[i];
                        }
                    }

                    if (Owner.activeCells < Owner.MAX_CELLS)
                    {
                        float globalErrorDiff = Math.Abs(localErrorSum - lastLocalErrorSum);
                        
                        if (localErrorSum > GLOBAL_E_TH && globalErrorDiff <= DIFF_TOLERANCE)
                        {
                            // determine the unit q with the maximum accumulated error
                            float maxError = 0.00f;
                            int q = -1;
                            for (int i = 0; i < Owner.MAX_CELLS; i++)
                            {
                                if (Owner.ActivityFlag.Host[i] == 1)
                                {
                                    if (Owner.LocalError.Host[i] >= maxError)
                                    {
                                        maxError = Owner.LocalError.Host[i];
                                        q = i;
                                    }
                                }
                            }

                            // determine among the neighbors of the q the unit f with the maximum accumulated error
                            Owner.ConnectionMatrix.SafeCopyToHost();
                            int f = -1;
                            maxError = 0.00f;
                            for (int i = 0; i < Owner.MAX_CELLS; i++)
                            {
                                if (Owner.ConnectionMatrix.Host[q * Owner.MAX_CELLS + i] == 1)
                                {
                                    if (Owner.ActivityFlag.Host[i] == 1)
                                    {
                                        if (Owner.LocalError.Host[i] >= maxError)
                                        {
                                            maxError = Owner.LocalError.Host[i];
                                            f = i;
                                        }
                                    }
                                }
                            }

                            // add a new unit r to the newtork and interpole its reference vector from q and f
                            int r = -1;
                            for (int i = 0; i < Owner.MAX_CELLS; i++)
                            {
                                if (Owner.ActivityFlag.Host[i] == 0)
                                {
                                    r = i;
                                    //Owner.ActivityFlag.Host[i] = 1;
                                    break;
                                }
                            }

                            m_InterpolateVectorKernel.m_kernel.SetupExecution(Owner.INPUT_SIZE
                                );

                            m_InterpolateVectorKernel.Run(r, q, f,
                                Owner.INPUT_SIZE,
                                Owner.ReferenceVector
                                );

                            // insert edges connecting the new unit r with units q and f and remove the original edge between q and f
                            // decrease the rror variables of q and f by a fraction ALFA
                            // interpolate the error variable of r from q and f
                            m_newNodeConnectionKernel.m_kernel.SetupExecution(1
                                );

                            m_newNodeConnectionKernel.Run(f, q, r,
                                Owner.ActivityFlag,
                                Owner.ConnectionMatrix,
                                Owner.ConnectionAge,
                                Owner.LocalError,
                                ALFA,
                                Owner.MAX_CELLS,
                                0.50f
                                );

                            Owner.activeCells++;
                        }
                    }
                    lastLocalErrorSum = localErrorSum;
                }
                */

                if (EQUILIBRIUM == Turning.On && Owner.activeCells < Owner.MAX_CELLS)
                {
                    float localErrorSum = 0.00f;
                    Owner.LocalError.SafeCopyToHost();
                    for (int i = 0; i < Owner.MAX_CELLS; i++)
                    {
                        if (Owner.ActivityFlag.Host[i] == 1)
                        {
                            localErrorSum += Owner.LocalError.Host[i];
                        }
                    }

                    float avgLocalError = localErrorSum / Owner.activeCells;

                    // avg local error is greater then threshold
                    if (avgLocalError > AVG_E_TH)
                    {
                        // determine the unit q with the maximum accumulated error
                        float maxError = 0.00f;
                        int q = -1;
                        for (int i = 0; i < Owner.MAX_CELLS; i++)
                        {
                            if (Owner.ActivityFlag.Host[i] == 1)
                            {
                                if (Owner.LocalError.Host[i] >= maxError)
                                {
                                    maxError = Owner.LocalError.Host[i];
                                    q = i;
                                }
                            }
                        }

                        // determine among the neighbors of the q the unit f with the maximum accumulated error
                        Owner.ConnectionMatrix.SafeCopyToHost();
                        int f = -1;
                        maxError = 0.00f;
                        for (int i = 0; i < Owner.MAX_CELLS; i++)
                        {
                            if (Owner.ConnectionMatrix.Host[q * Owner.MAX_CELLS + i] == 1)
                            {
                                if (Owner.ActivityFlag.Host[i] == 1)
                                {
                                    if (Owner.LocalError.Host[i] >= maxError)
                                    {
                                        maxError = Owner.LocalError.Host[i];
                                        f = i;
                                    }
                                }
                            }
                        }

                        // add a new unit r to the newtork and interpole its reference vector from q and f
                        int r = -1;
                        for (int i = 0; i < Owner.MAX_CELLS; i++)
                        {
                            if (Owner.ActivityFlag.Host[i] == 0)
                            {
                                r = i;
                                //Owner.ActivityFlag.Host[i] = 1;
                                break;
                            }
                        }

                        m_InterpolateVectorKernel.SetupExecution(Owner.INPUT_SIZE
                            );

                        m_InterpolateVectorKernel.Run(r, q, f,
                            Owner.INPUT_SIZE,
                            Owner.ReferenceVector
                            );

                        // insert edges connecting the new unit r with units q and f and remove the original edge between q and f
                        // decrease the rror variables of q and f by a fraction ALFA
                        // interpolate the error variable of r from q and f
                        m_newNodeConnectionKernel.SetupExecution(1
                            );

                        m_newNodeConnectionKernel.Run(f, q, r,
                            Owner.ActivityFlag,
                            Owner.ConnectionMatrix,
                            Owner.ConnectionAge,
                            Owner.LocalError,
                            ALFA,
                            Owner.MAX_CELLS,
                            0.50f
                            );

                        Owner.activeCells++;
                    }
                }

                
                if (VECTOR_OUTLIER == Turning.On && Owner.activeCells < Owner.MAX_CELLS)
                {
                    // compute distance between s1 and s2
                    m_twoNodesDifferenceKernel.SetupExecution(Owner.INPUT_SIZE
                        );

                    m_twoNodesDifferenceKernel.Run(Owner.s1,
                        Owner.s2,
                        Owner.INPUT_SIZE,
                        Owner.ReferenceVector,
                        Owner.TwoNodesDifference
                        );

                    m_twoNodesDistanceKernel.SetupExecution(1
                        );

                    m_twoNodesDistanceKernel.Run(Owner.TwoNodesDifference,
                        Owner.TwoNodesDistance,
                        Owner.INPUT_SIZE
                        );

                    Owner.TwoNodesDistance.SafeCopyToHost();
                    
                    // compute distance between s1 and INPUT
                    Owner.Distance.SafeCopyToHost();
                    float s1InputDistance = Owner.Distance.Host[Owner.s1];

                    // if d(INPUT,s1) > d(s1,s2) add outlier to the gas
                    if (s1InputDistance > DISTANCE * Owner.TwoNodesDistance.Host[0])
                    {
                        // find index for new gas cell
                        int r = -1;
                        for (int i = 0; i < Owner.MAX_CELLS; i++)
                        {
                            if (Owner.ActivityFlag.Host[i] == 0)
                            {
                                r = i;
                                //Owner.ActivityFlag.Host[i] = 1;
                                break;
                            }
                        }

                        m_copyVectorKernel.SetupExecution(Owner.INPUT_SIZE
                            );

                        m_copyVectorKernel.Run(Owner.Input,
                            0,
                            Owner.ReferenceVector,
                            r * Owner.INPUT_SIZE,
                            Owner.INPUT_SIZE
                            );


                        m_addAndRefreshConnectionKernel.SetupExecution(1
                            );

                        m_addAndRefreshConnectionKernel.Run(Owner.s1,
                            r,
                            Owner.ActivityFlag,
                            Owner.ConnectionMatrix,
                            Owner.ConnectionAge,
                            Owner.MAX_CELLS
                            );

                        m_addAndRefreshConnectionKernel.Run(Owner.s2,
                            r,
                            Owner.ActivityFlag,
                            Owner.ConnectionMatrix,
                            Owner.ConnectionAge,
                            Owner.MAX_CELLS
                            );

                        Owner.LocalError.GetDevice(Owner).CopyToDevice(0.00f, r * sizeof(float));
                        Owner.activeCells++;
                    }

                }

                if (INCREMENTAL == Turning.On && Owner.activeCells < Owner.MAX_CELLS)
                {

                    if (SimulationStep % LAMBDA == 0)
                    {
                        // determine the unit q with the maximum accumulated error
                        Owner.LocalError.SafeCopyToHost();
                        float maxError = 0.00f;
                        int q = -1;
                        for (int i = 0; i < Owner.MAX_CELLS; i++)
                        {
                            if (Owner.ActivityFlag.Host[i] == 1)
                            {
                                if (Owner.LocalError.Host[i] >= maxError)
                                {
                                    maxError = Owner.LocalError.Host[i];
                                    q = i;
                                }
                            }
                        }

                        // determine among the neighbors of the q the unit f with the maximum accumulated error
                        Owner.ConnectionMatrix.SafeCopyToHost();
                        int f = -1;
                        maxError = 0.00f;
                        for (int i = 0; i < Owner.MAX_CELLS; i++)
                        {
                            if (Owner.ConnectionMatrix.Host[q * Owner.MAX_CELLS + i] == 1)
                            {
                                if (Owner.ActivityFlag.Host[i] == 1)
                                {
                                    if (Owner.LocalError.Host[i] >= maxError)
                                    {
                                        maxError = Owner.LocalError.Host[i];
                                        f = i;
                                    }
                                }
                            }
                        }

                        // add a new unit r to the newtork and interpole its reference vector from q and f
                        int r = -1;
                        for (int i = 0; i < Owner.MAX_CELLS; i++)
                        {
                            if (Owner.ActivityFlag.Host[i] == 0)
                            {
                                r = i;
                                //Owner.ActivityFlag.Host[i] = 1;
                                break;
                            }
                        }

                        m_InterpolateVectorKernel.SetupExecution(Owner.INPUT_SIZE
                            );

                        m_InterpolateVectorKernel.Run(r, q, f,
                            Owner.INPUT_SIZE,
                            Owner.ReferenceVector
                            );

                        // insert edges connecting the new unit r with units q and f and remove the original edge between q and f
                        // decrease the rror variables of q and f by a fraction ALFA
                        // interpolate the error variable of r from q and f
                        m_newNodeConnectionKernel.SetupExecution(1
                            );

                        m_newNodeConnectionKernel.Run(f, q, r,
                            Owner.ActivityFlag,
                            Owner.ConnectionMatrix,
                            Owner.ConnectionAge,
                            Owner.LocalError,
                            ALFA,
                            Owner.MAX_CELLS,
                            0.50f
                            );

                        Owner.activeCells++;
                    }
                }
            }

            

            
        }
    }

    /// <summary>
    /// Decrease accumulated error of each cell by the factor BETA
    /// </summary>
    [Description("Decrease error and utility variable"), MyTaskInfo(OneShot = false)]
    public class MyDecreaseErrorAndUtilityTask : MyTask<MyGrowingNeuralGasNode>
    {
        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 0.0005f), YAXElementFor("Structure")]
        public float BETA { get; set; }

        private MyCudaKernel m_decreaseLocalErrorAndUtilityKernel;

        public override void Init(int nGPU)
        {
            m_decreaseLocalErrorAndUtilityKernel = MyKernelFactory.Instance.Kernel(nGPU, @"GrowingNeuralGas\GrowingNeuralGasKernel", "DecreaseErrorAndUtilityKernel");
        }
        
        public override void Execute()
        {
            m_decreaseLocalErrorAndUtilityKernel.SetupExecution(Owner.MAX_CELLS
                );

            m_decreaseLocalErrorAndUtilityKernel.Run(Owner.LocalError,
                Owner.Utility,
                Owner.ActivityFlag,
                Owner.MAX_CELLS,
                BETA
                );
        }

    }
}
