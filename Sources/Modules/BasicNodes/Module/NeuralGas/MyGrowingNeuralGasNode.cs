using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.NeuralGas
{
    /// <author>GoodAI</author>
    /// <meta>rb</meta>
    /// <status>Working</status>
    /// <summary>Growing neural gas implementation with various growing mechanisms</summary>
    /// <description>
    /// Parameters:<br />
    /// <ul>
    /// <li>MAX_CELLS: maximum possible number of active neural gas cells</li>
    /// <li>INIT_LIVE_CELLS: the initiali active cell of neural gas</li>
    /// <li>INPUT_SIZE: size of the input vector, also the length of neural gas vector, set automatically, read only</li>
    /// </ul>
    /// </description>
    public class MyGrowingNeuralGasNode : MyWorkingNode
    {
        [MyBrowsable, Category("Input interface")]
        [YAXSerializableField(DefaultValue = 100), YAXElementFor("Structure")]
        public int MAX_CELLS { get; set; }

        [MyBrowsable, Category("Input interface")]
        [YAXSerializableField(DefaultValue = 2), YAXElementFor("Structure")]
        public int INIT_LIVE_CELLS { get; set; }

        [MyBrowsable, Category("Input interface")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int INPUT_SIZE { get; private set; }

        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> OutputOne
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> WinnerOne
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> OutputTwo
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> WinnerTwo
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        public int activeCells;
        public int s1, s2;
        
        // MEMORY BLOCKS
        [MyPersistable]
        public MyMemoryBlock<float> ReferenceVector { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<int> ConnectionMatrix { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<int> ActivityFlag { get; private set; }

        public MyMemoryBlock<int> ConnectionAge { get; private set; }
        public MyMemoryBlock<float> Distance { get; private set; }
        public MyMemoryBlock<float> LocalError { get; private set; }
        public MyMemoryBlock<float> Utility { get; private set; }

        public MyMemoryBlock<float> Difference { get; private set; }
        public MyMemoryBlock<float> DimensionWeight { get; private set; }



        // outlier
        public MyMemoryBlock<float> TwoNodesDifference { get; private set; }
        public MyMemoryBlock<float> TwoNodesDistance { get; private set; }
        
        public MyMemoryBlock<int> NeuronAge { get; private set; }
        public MyMemoryBlock<int> WinningCount { get; private set; }

        //conscience
        public MyMemoryBlock<float> WinningFraction { get; private set; }
        public MyMemoryBlock<float> BiasTerm { get; private set; }
        public MyMemoryBlock<float> BiasedDistance { get; private set; }

        // TASKS
        public MyInitGrowingNeuralGasNodeTask InitGrowingNeuralGas {get; private set;}
        public MyFindWinnersTask Findwinners {get; private set;}
        public MyAdaptWinningFractionTask AdaptWinningFraction { get; private set; }
        public MyBiasTermTask ComputeBiasTerm { get; private set; }
        public MyFindConsciousWinnersTask FindConsciousWinners { get; private set; }
        public MySendDataToOutputTask SendDataToOutput { get; private set; }
        public MyAddLocalErrorAndUtilityTask AddLocalAndUtilityError { get; private set; }
        public MyCreateConnecionTask CreateConnection {get; private set;}
        public MyAdaptRefVectorTask AdaptRefVector {get; private set;}
        public MyIncrementConnectionAgeTask IncrementConnectionAge {get; private set;}
        public MyRemoveConnsAndCellsTask RemoveConnsAndCells {get; private set;}
        public MyAddNewNodeTask AddNewNodeTask {get; private set;}
        public MyDecreaseErrorAndUtilityTask DecreaseErrorTask {get; private set;}

        public override void UpdateMemoryBlocks()
        {
            INPUT_SIZE = Input == null ? 1 : Input.Count;
            ReferenceVector.Count = INPUT_SIZE * MAX_CELLS;

            Distance.Count = MAX_CELLS;

            // TO DO: DO NOT USE FULL ADJACENCY MATRIX
            ConnectionMatrix.Count = MAX_CELLS * MAX_CELLS;
            ConnectionAge.Count = ConnectionMatrix.Count;
            ActivityFlag.Count = MAX_CELLS;
            LocalError.Count = MAX_CELLS;
            

            Difference.Count = ReferenceVector.Count;
            
            TwoNodesDifference.Count = INPUT_SIZE;
            TwoNodesDistance.Count = 1;

            WinningFraction.Count = MAX_CELLS;
            BiasTerm.Count = MAX_CELLS;
            BiasedDistance.Count = MAX_CELLS;

            OutputOne.Count = INPUT_SIZE;
            OutputOne.ColumnHint = Input == null ? 1: Input.ColumnHint;
            WinnerOne.Count = MAX_CELLS;
            OutputTwo.Count = INPUT_SIZE;
            OutputTwo.ColumnHint = Input == null ? 1 : Input.ColumnHint;
            WinnerTwo.Count = MAX_CELLS;

            NeuronAge.Count = MAX_CELLS;
            WinningCount.Count = MAX_CELLS;

            DimensionWeight.Count = INPUT_SIZE;

            Utility.Count = MAX_CELLS;
        }

        public override string Description
        {
            get
            {
                return "GNG";
            }
        }
    }
}
