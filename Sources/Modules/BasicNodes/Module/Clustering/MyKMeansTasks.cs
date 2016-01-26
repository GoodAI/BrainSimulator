using GoodAI.Core;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using ManagedCuda;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Clustering
{
    [Description("Init K-means node"), MyTaskInfo(OneShot = true)]
    public class MyInitKMeansNodeTask : MyTask<MyKMeansNode>
    {
        public override void Init(int nGPU)
        {
            
        }

        public override void Execute()
        {
            
        }
    }

    [Description("Cluster input"), MyTaskInfo(OneShot = false)]
    public class MyClusterTask : MyTask<MyKMeansNode>
    {
        [MyBrowsable, Category("Init")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float X_MIN { get; set; }

        [MyBrowsable, Category("Init")]
        [YAXSerializableField(DefaultValue = 1.00f), YAXElementFor("Structure")]
        public float X_MAX { get; set; }

        [MyBrowsable, Category("Init")]
        [YAXSerializableField(DefaultValue = 0.00f), YAXElementFor("Structure")]
        public float Y_MIN { get; set; }

        [MyBrowsable, Category("Init")]
        [YAXSerializableField(DefaultValue = 1.00f), YAXElementFor("Structure")]
        public float Y_MAX { get; set; }

        [MyBrowsable, Category("Clustering")]
        [YAXSerializableField(DefaultValue = 35), YAXElementFor("Structure")]
        public int STEPS { get; set; }

        private MyCudaKernel m_initCentroidsKernel;
        private MyCudaKernel m_computeEuklidianDistancesKernel;
        private MyCudaKernel m_findNearestCentroidKernel;
        private MyCudaKernel m_sumNewCentroidCoordinatesKernel;
        private MyCudaKernel m_avgCentroidCoordinatesKernel;

        private MyCudaKernel m_copyInputToVisFieldKernel;
        private MyCudaKernel m_markCentroidsKernel;

        private MyReductionKernel<float> m_reduction;

        private CudaDeviceVariable<int> m_d_changeFlag;
        private int[] m_h_changeFlag;

        public override void Init(int nGPU)
        {
            m_initCentroidsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Clustering\KMeansKernel", "InitCentroidsKernel");
            m_computeEuklidianDistancesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Clustering\KMeansKernel", "ComputeEuklidianDistancesKernel");
            m_findNearestCentroidKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Clustering\KMeansKernel", "FindNearestCentroidKernel");
            m_sumNewCentroidCoordinatesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Clustering\KMeansKernel", "SumNewCentroidCoordinatesKernel");
            m_avgCentroidCoordinatesKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Clustering\KMeansKernel", "AvgCentroidCoordinatesKernel");

            m_copyInputToVisFieldKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Clustering\KMeansKernel", "CopyInputToVisFieldKernel");
            m_markCentroidsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Clustering\KMeansKernel", "MarkCentroidsKernel");

            m_reduction = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_Sum_f);

            m_d_changeFlag = new CudaDeviceVariable<int>(1);
            m_h_changeFlag = new int[1];
        }

        public override void Execute()
        {
            // inint centroid coordinates
            if (SimulationStep == 0)
            {
                MyKernelFactory.Instance.GetRandDevice(Owner).GenerateUniform(Owner.RandomNumbers.GetDevice(Owner));
                m_initCentroidsKernel.SetupExecution(Owner.CLUSTERS
                    );
                m_initCentroidsKernel.Run(Owner.CentroidCoordinates,
                    Owner.RandomNumbers,
                    X_MIN,
                    X_MAX,
                    Y_MIN,
                    Y_MAX,
                    Owner.CLUSTERS
                    );
            }
            m_computeEuklidianDistancesKernel.SetupExecution(Owner.INPUT_SIZE
                );

            m_findNearestCentroidKernel.SetupExecution(Owner.INPUT_SIZE
                );

            m_sumNewCentroidCoordinatesKernel.SetupExecution(Owner.INPUT_SIZE
                );

            m_avgCentroidCoordinatesKernel.SetupExecution(Owner.CLUSTERS * 2
                    );

            for (int s = 0; s < STEPS; s++)
            {
                m_computeEuklidianDistancesKernel.Run(Owner.Input,
                    Owner.IMG_WIDTH,
                    Owner.IMG_HEIGHT,
                    Owner.CentroidCoordinates,
                    Owner.DistanceMatrix,
                    Owner.CLUSTERS,
                    Owner.INPUT_SIZE
                    );

                m_findNearestCentroidKernel.Run(Owner.DistanceMatrix,
                    Owner.NearestCentroid,
                    Owner.CLUSTERS,
                    Owner.INPUT_SIZE
                    );

                Owner.CentroidCoordinates.Fill(0.00f);
                Owner.PointsWeight.Fill(0.00f);

                m_sumNewCentroidCoordinatesKernel.Run(Owner.Input,
                    Owner.IMG_WIDTH,
                    Owner.IMG_HEIGHT,
                    Owner.CentroidCoordinates,
                    Owner.NearestCentroid,
                    Owner.PointsWeight,
                    Owner.INPUT_SIZE
                    );

                m_avgCentroidCoordinatesKernel.Run(Owner.CentroidCoordinates,
                    Owner.PointsWeight,
                    Owner.INPUT_SIZE,
                    Owner.CLUSTERS
                    );
            }

            // copy input to the visualization field with the centroid coordinates
            m_copyInputToVisFieldKernel.SetupExecution(Owner.INPUT_SIZE
                );
            m_copyInputToVisFieldKernel.Run(Owner.Input,
                Owner.VisField,
                Owner.INPUT_SIZE
                );

            m_markCentroidsKernel.SetupExecution(Owner.CLUSTERS
                );

            m_markCentroidsKernel.Run(Owner.CentroidCoordinates,
                Owner.VisField,
                Owner.IMG_WIDTH,
                Owner.IMG_HEIGHT,
                Owner.CLUSTERS
                );

        }
    }
}
