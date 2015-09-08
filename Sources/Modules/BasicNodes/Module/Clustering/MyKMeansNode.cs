using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Clustering
{
    /// <author>Radoslav Bielek</author>
    /// <status>WIP</status>
    /// <summary>Not finished version of the K-means algorithm</summary>
    /// <description></description>
    public class MyKMeansNode : MyWorkingNode
    {
        [MyBrowsable, Category("Parameters")]
        [YAXSerializableField(DefaultValue = 5), YAXElementFor("Structure")]
        public int CLUSTERS { get; set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int INPUT_SIZE { get; private set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = InputMode.GrayscaleIMG), YAXElementFor("Structure")]
        public InputMode INPUT_MODE { get; private set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int IMG_WIDTH { get; private set; }

        [MyBrowsable, Category("Input")]
        [YAXSerializableField(DefaultValue = 1), YAXElementFor("Structure")]
        public int IMG_HEIGHT { get; private set; }
        
        public enum InputMode
        {
            GrayscaleIMG
        }

        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }


        // MEMORY BLOCKS
        public MyMemoryBlock<float> CentroidCoordinates { get; private set; }
        public MyMemoryBlock<float> DistanceMatrix { get; private set; }
        public MyMemoryBlock<int> NearestCentroid { get; private set; }

        public MyMemoryBlock<float> RandomNumbers { get; private set; }

        public MyMemoryBlock<uint> ClusterColors { get; private set; }
        public MyMemoryBlock<float> PointsWeight { get; private set; }

        public MyMemoryBlock<float> VisField { get; private set; }


        // TASKS
        public MyInitKMeansNodeTask InitKMeansNode { get; private set; }
        public MyClusterTask ClusterTask { get; private set; }

        public override void UpdateMemoryBlocks()
        {
            INPUT_SIZE = Input == null ? 1 : Input.Count;

            IMG_WIDTH = Input == null ? 1 : Input.ColumnHint;
            IMG_HEIGHT = Input == null ? 1 : INPUT_SIZE / IMG_WIDTH;

            ClusterColors.Count = Input == null ? 1 : Input.Count;
            ClusterColors.ColumnHint = Input == null ? 1 : Input.ColumnHint;

            CentroidCoordinates.Count = CLUSTERS * 2;
            RandomNumbers.Count = CLUSTERS * 2;

            DistanceMatrix.Count = CLUSTERS * INPUT_SIZE;
            //DistanceMatrix.ColumnHint = INPUT_SIZE;

            NearestCentroid.Count = INPUT_SIZE;


            PointsWeight.Count = CLUSTERS;

            VisField.Count = INPUT_SIZE;
            VisField.ColumnHint = IMG_WIDTH;

        }

    }
}
