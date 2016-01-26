using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using ManagedCuda.VectorTypes;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Retina
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    ///<status>Working (K-Means only)</status>
    ///<summary>Analyzes visual input through growing K-Means clustering.
    ///Clusters are iterated and current cluster data (position and scale) is sent to the output.</summary>
    ///<description></description>
    public class MyPupilControl : MyWorkingNode
    {
        [MyInputBlock]
        public MyMemoryBlock<float> AttentionMap
        {
            get { return GetInput(0); }
        }

        [MyOutputBlock]
        public MyMemoryBlock<float> PupilControl
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public enum ControlMethod
        {
            SimpleMean,
            K_Means
        };

        [MyBrowsable, Category("Params")]
        [YAXSerializableField(DefaultValue = ControlMethod.SimpleMean)]
        public ControlMethod Method { get; set; }

        public MyMemoryBlock<float> ReductionSources { get; protected set; }
        public MyMemoryBlock<float> Statistics { get; protected set; }

        public MyMemoryBlock<float> Centroids { get; protected set; }
        public MyMemoryBlock<float> CentroidsDBI { get; protected set; }
        public MyMemoryBlock<float> CentroidImportance { get; protected set; }

        public MyMemoryBlock<float> ControlValues { get; protected set; }

        public MyPupilControlTask MovePupil { get; protected set; }

        protected int m_sumsPerItem;
        protected int m_centroidsCount = 5;
        internal const int CENTROID_FIELDS = 6;
        internal const int MAX_CENTROIDS = 256;

        public int CentroidsCount
        {
            get { return m_centroidsCount; }
        }

        public override void UpdateMemoryBlocks()
        {
            PupilControl.Count = 3;

            if (Method == ControlMethod.SimpleMean)
            {
                m_sumsPerItem = 4;
                Statistics.Count = 5;
                Centroids.Count = 0;
                CentroidImportance.Count = 0;
                ControlValues.Count = 0;
                CentroidsDBI.Count = 0;

                if (AttentionMap != null)
                {
                    ReductionSources.Count = AttentionMap.Count * m_sumsPerItem;
                    ReductionSources.ColumnHint = AttentionMap.ColumnHint;
                }
            }
            else
            {
                m_sumsPerItem = 5;
                Statistics.Count = MAX_CENTROIDS * m_sumsPerItem + 1;
                Centroids.Count = MAX_CENTROIDS * CENTROID_FIELDS;
                CentroidImportance.Count = MAX_CENTROIDS;
                ControlValues.Count = 3;
                CentroidsDBI.Count = MAX_CENTROIDS * MAX_CENTROIDS;
                CentroidsDBI.ColumnHint = MAX_CENTROIDS;

                if (AttentionMap != null)
                {
                    ReductionSources.Count = AttentionMap.Count * m_sumsPerItem * MAX_CENTROIDS;
                    ReductionSources.ColumnHint = AttentionMap.ColumnHint;
                }
            }
        }

        [Description("Pupil Control")]
        public class MyPupilControlTask : MyTask<MyPupilControl>
        {
            private MyCudaKernel m_kernel;
        
            [MyBrowsable, Category("Simple Control")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float MoveFactor { get; set; }

            [MyBrowsable, Category("Simple Control")]
            [YAXSerializableField(DefaultValue = 3.0f)]
            public float ScaleFactor { get; set; }

            [MyBrowsable, Category("Simple Control")]
            [YAXSerializableField(DefaultValue = 0.1f)]
            public float ScaleBase { get; set; }

            [MyBrowsable, Category("K-Means Control")]
            [YAXSerializableField(DefaultValue = 1.0f)]
            public float LearningRate { get; set; }

            [MyBrowsable, Category("K-Means Control")]
            [YAXSerializableField(DefaultValue = 20.0f)]
            public float FocusInterval { get; set; }

            private int m_imageWidth, m_imageHeight;
            private MyReductionKernel<float> m_reduction_kernel;
            private MyCudaKernel m_finalize_kernel;
            private MyCudaKernel m_eye_kernel;
            private MyCudaKernel m_dbiKernel;
            private MyCudaKernel m_1meansKernel;
            private MyCudaKernel m_2meansKernel;
            private MyCudaKernel m_setKernel;

            private MyCudaKernel m_sumKernel;

            private Random random;

            public override void Init(int nGPU)
            {
                random = new Random();

                if (Owner.Method == ControlMethod.SimpleMean)
                {
                    m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "PrepareMeanStdDev");
                    m_reduction_kernel = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_Sum_f);
                    m_finalize_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "FinalizeMeanStdDev");

                    m_eye_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Retina\FindBrightSpotKernel", "ApplyEyeMovement");

                    m_imageWidth = Owner.AttentionMap.ColumnHint;
                    m_imageHeight = Owner.AttentionMap.Count / m_imageWidth;

                    m_finalize_kernel.SetupExecution(1);
                }
                else
                {
                    m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "PrepareK_Means");
                    m_reduction_kernel = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_Sum_f);
                    m_sumKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "SumCentroids");                    
                    m_finalize_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "FinalizeK_Means");

                    m_dbiKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "EvaluateDBI");
                    m_1meansKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "Prepare_1_MeansForJoin");
                    m_2meansKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\Statistics", "Prepare_2_MeansForDivision");

                    m_eye_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Retina\FindBrightSpotKernel", "ApplyEyeMovement");
                    m_setKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\SetKernel");

                    m_imageWidth = Owner.AttentionMap.ColumnHint;
                    m_imageHeight = Owner.AttentionMap.Count / m_imageWidth;
                }

                m_kernel.SetupExecution(Owner.AttentionMap.Count);
                m_1meansKernel.SetupExecution(Owner.AttentionMap.Count);
                m_2meansKernel.SetupExecution(Owner.AttentionMap.Count);

                m_eye_kernel.SetupExecution(1);

                Owner.m_centroidsCount = 5;
            }

            public override void Execute()
            {
                int usedSpace = (2 * Owner.CentroidsCount + 1) * Owner.m_sumsPerItem;

                Owner.PupilControl.Fill(0);
                Owner.Statistics.Fill(0);

                m_setKernel.SetupExecution(usedSpace * Owner.AttentionMap.Count);
                m_setKernel.Run(Owner.ReductionSources, 0, 0, usedSpace * Owner.AttentionMap.Count);

                //m_setKernel.m_kernel.SetupExecution(usedSpace);
                //m_setKernel.Run(Owner.Statistics, 0, 0, usedSpace);                

                if (Owner.Method == ControlMethod.SimpleMean)
                {
                    ExecuteSimple();
                }
                else
                {
                    ExecuteK_Means();
                }                
            }

            private void ExecuteK_Means()
            {
                //init all centroids into small circle around the center
                if (SimulationStep == 0)
                {
                    for (int i = 0; i < Owner.m_centroidsCount; i++)
                    {
                        double angle = (double)i / Owner.m_centroidsCount * Math.PI * 2;
                        double radius = 0.2f;

                        double x = radius * Math.Cos(angle);
                        double y = radius * Math.Sin(angle);

                        Owner.Centroids.Host[CENTROID_FIELDS * i] = (float)x;
                        Owner.Centroids.Host[CENTROID_FIELDS * i + 1] = (float)y;

                        Owner.Centroids.Host[CENTROID_FIELDS * i + 2] = 0;
                        Owner.Centroids.Host[CENTROID_FIELDS * i + 3] = 0;
                    }

                    Owner.Centroids.SafeCopyToDevice();
                    Owner.ControlValues.SafeCopyToDevice();
                }
                //process K-Means
                else
                {                    
                    //assign pixels to centroids
                    m_kernel.Run(Owner.AttentionMap, Owner.Centroids, Owner.m_centroidsCount, Owner.ReductionSources, m_imageWidth, m_imageHeight);                    

                    //sum centroid values for mean & stdDev
                    for (int i = 0; i < Owner.m_centroidsCount; i++)
                    {
                        for (int j = 0; j < Owner.m_sumsPerItem; j++)
                        {
                            //ZXC m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count,
                            //    Owner.m_sumsPerItem * i + j, i * Owner.AttentionMap.Count * Owner.m_sumsPerItem + j * Owner.AttentionMap.Count, 1, /* distributed: */ 0);*/
                            m_reduction_kernel.outOffset = Owner.m_sumsPerItem * i + j;
                            m_reduction_kernel.inOffset = i * Owner.AttentionMap.Count * Owner.m_sumsPerItem + j * Owner.AttentionMap.Count;
                            m_reduction_kernel.size = Owner.AttentionMap.Count;
                            m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);
                        }
                    }

                    //sum all weights (denominator)
                    //ZXC m_reduction_kernel.Run(Owner.Statistics, Owner.AttentionMap, Owner.AttentionMap.Count, Owner.m_sumsPerItem * Owner.m_centroidsCount, 0, 1, /* distributed: */ 0);
                    m_reduction_kernel.outOffset = Owner.m_sumsPerItem * Owner.m_centroidsCount;
                    m_reduction_kernel.inOffset = 0;
                    m_reduction_kernel.size = Owner.AttentionMap.Count;
                    m_reduction_kernel.Run(Owner.Statistics, Owner.AttentionMap);

                    //evaluate new mean & stdDev for all centroids
                    m_finalize_kernel.SetupExecution(Owner.m_centroidsCount);
                    m_finalize_kernel.Run(Owner.Centroids, Owner.m_centroidsCount, Owner.Statistics, LearningRate, 0);

                    //TODO: move it to GPU  
                    Owner.Centroids.SafeCopyToHost();
                    Owner.ControlValues.SafeCopyToHost();
                    Owner.Statistics.SafeCopyToHost();

                    FocusToCentroid();

                    if (SimulationStep % 2 == 0 && Owner.m_centroidsCount > 1)
                    {
                        DeleteUnusedCentroids();
                    }
                    else if (SimulationStep % 4 == 1 && Owner.m_centroidsCount < MAX_CENTROIDS * 0.125)
                    {
                        PerformSplit();
                    }
                    else if (Owner.m_centroidsCount > 1 && SimulationStep % 10 == 3)
                    {
                        PerformJoin();
                    }

                    int unusedCount = (MAX_CENTROIDS - Owner.m_centroidsCount) * CENTROID_FIELDS;
                    m_setKernel.SetupExecution(unusedCount);
                    m_setKernel.Run(Owner.Centroids, Owner.m_centroidsCount * CENTROID_FIELDS, float.PositiveInfinity, unusedCount);

                    Owner.Centroids.SafeCopyToHost();

                    Owner.PupilControl.SafeCopyToDevice();
                    Owner.CentroidImportance.SafeCopyToDevice();
                    Owner.ControlValues.SafeCopyToDevice();                 
                }
            }

            private void DeleteUnusedCentroids()
            {
                int i = 0;

                while (i < Owner.m_centroidsCount)
                {
                    if (float.IsInfinity(Owner.Centroids.Host[i * CENTROID_FIELDS + 5]))
                    {
                        if (i != Owner.m_centroidsCount - 1)
                        {
                            Owner.Centroids.CopyToMemoryBlock(Owner.Centroids, (Owner.m_centroidsCount - 1) * CENTROID_FIELDS, i * CENTROID_FIELDS, CENTROID_FIELDS);
                        }
                        Owner.m_centroidsCount--;
                    }
                    i++;
                }
            }

            private void PerformSplit()
            {
                //generate random spit for every centroid
                for (int i = 0; i < Owner.m_centroidsCount; i++)
                {
                    float2 c_src = new float2(Owner.Centroids.Host[i * CENTROID_FIELDS], Owner.Centroids.Host[i * CENTROID_FIELDS + 1]);

                    float vx = Owner.Centroids.Host[i * CENTROID_FIELDS + 2];
                    float vy = Owner.Centroids.Host[i * CENTROID_FIELDS + 3];

                    double dir = random.NextDouble() * Math.PI * 2;

                    int c_n1i = Owner.m_centroidsCount + 2 * i;
                    int c_n2i = Owner.m_centroidsCount + 2 * i + 1;

                    float2 c_n1 = new float2((float)(vx * 0.5 * Math.Cos(dir)), (float)(vy * 0.5 * Math.Sin(dir)));
                    float2 c_n2 = c_n1 * -1;

                    c_n1 += c_src;
                    c_n2 += c_src;

                    Owner.Centroids.Host[c_n1i * CENTROID_FIELDS] = c_n1.x;
                    Owner.Centroids.Host[c_n1i * CENTROID_FIELDS + 1] = c_n1.y;

                    Owner.Centroids.Host[c_n2i * CENTROID_FIELDS] = c_n2.x;
                    Owner.Centroids.Host[c_n2i * CENTROID_FIELDS + 1] = c_n2.y;
                }

                Owner.Centroids.SafeCopyToDevice();

                //try 2-Means for generated pairs
                for (int i = 0; i < Owner.m_centroidsCount; i++)
                {
                    int c_n1i = Owner.m_centroidsCount + 2 * i;
                    int c_n2i = Owner.m_centroidsCount + 2 * i + 1;

                    //prepare 2-Means
                    m_2meansKernel.Run(Owner.AttentionMap, Owner.Centroids, c_n1i, c_n2i, i,
                        Owner.ReductionSources, m_imageWidth, m_imageHeight);

                    //do reductions
                    for (int k = 0; k < Owner.m_sumsPerItem; k++)
                    {
                        //m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count,
                        //           c_n1i * Owner.m_sumsPerItem + k,
                        //           c_n1i * Owner.AttentionMap.Count * Owner.m_sumsPerItem + k * Owner.AttentionMap.Count, 1, /* distributed: */ 0);
                        m_reduction_kernel.size = Owner.AttentionMap.Count;
                        m_reduction_kernel.outOffset = c_n1i * Owner.m_sumsPerItem + k;
                        m_reduction_kernel.inOffset = c_n1i * Owner.AttentionMap.Count * Owner.m_sumsPerItem + k * Owner.AttentionMap.Count;
                        m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);

                        //m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count,
                        //           c_n2i * Owner.m_sumsPerItem + k,
                        //           c_n2i * Owner.AttentionMap.Count * Owner.m_sumsPerItem + k * Owner.AttentionMap.Count, 1, /* distributed: */ 0);
                        m_reduction_kernel.size = Owner.AttentionMap.Count;
                        m_reduction_kernel.outOffset = c_n2i * Owner.m_sumsPerItem + k;
                        m_reduction_kernel.inOffset = c_n2i * Owner.AttentionMap.Count * Owner.m_sumsPerItem + k * Owner.AttentionMap.Count;
                        m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);
                    }
                }

                //finalize 2-Means
                m_finalize_kernel.SetupExecution(Owner.m_centroidsCount * 2);
                m_finalize_kernel.Run(Owner.Centroids, Owner.m_centroidsCount * 2, Owner.Statistics, LearningRate, Owner.m_centroidsCount);

                Owner.Centroids.SafeCopyToHost();

                //iterate candidates for split and find the maximum gain
                int numOfSplits = 0;
                for (int i = 0; i < Owner.m_centroidsCount; i++)
                {
                    int c_n1i = Owner.m_centroidsCount + 2 * i;
                    int c_n2i = Owner.m_centroidsCount + 2 * i + 1;

                    float2 c_n1 = new float2(Owner.Centroids.Host[c_n1i * CENTROID_FIELDS], Owner.Centroids.Host[c_n1i * CENTROID_FIELDS + 1]);
                    float2 c_n2 = new float2(Owner.Centroids.Host[c_n2i * CENTROID_FIELDS], Owner.Centroids.Host[c_n2i * CENTROID_FIELDS + 1]);
                    float2 c_n12 = c_n1 - c_n2;
                    float c_n12_l = (float)Math.Sqrt(c_n12.x * c_n12.x + c_n12.y * c_n12.y);

                    float c_n1_dbi = Owner.Centroids.Host[c_n1i * CENTROID_FIELDS + 5];
                    float c_n2_dbi = Owner.Centroids.Host[c_n2i * CENTROID_FIELDS + 5];

                    float c_src_dbi = Owner.Centroids.Host[i * CENTROID_FIELDS + 5];
                    float c_n12_dbi = 0.5f * (c_n1_dbi + c_n2_dbi) / c_n12_l;

                    if (c_n12_dbi < c_src_dbi)
                    {
                        //replace the source centroid to the first child
                        Owner.Centroids.CopyToMemoryBlock(Owner.Centroids, c_n1i * CENTROID_FIELDS, i * CENTROID_FIELDS, CENTROID_FIELDS);

                        //add the second child to the end
                        Owner.Centroids.CopyToMemoryBlock(Owner.Centroids, c_n2i * CENTROID_FIELDS,
                            (Owner.m_centroidsCount + numOfSplits) * CENTROID_FIELDS, CENTROID_FIELDS);

                        numOfSplits++;
                    }
                    //MyLog.DEBUG.WriteLine("Split candidates: " + i + ". = (" + c_src_dbi + " -> " + c_n12_dbi + ")");
                }

                Owner.m_centroidsCount += numOfSplits;
            }

            private void PerformJoin()
            {
                //Evaluate all DBI combinations
                m_dbiKernel.SetupExecution(Owner.m_centroidsCount * Owner.m_centroidsCount);
                m_dbiKernel.Run(Owner.Centroids, Owner.CentroidsDBI, Owner.m_centroidsCount);

                //Try join for all centroid pairs...
                int cIndex = 0;
                for (int j = 0; j < Owner.m_centroidsCount; j++)
                {
                    for (int i = j + 1; i < Owner.m_centroidsCount; i++)
                    {
                        float c_i_x = Owner.Centroids.Host[i * CENTROID_FIELDS];
                        float c_i_y = Owner.Centroids.Host[i * CENTROID_FIELDS + 1];
                        float c_i_vx = Owner.Centroids.Host[i * CENTROID_FIELDS + 2];
                        float c_i_vy = Owner.Centroids.Host[i * CENTROID_FIELDS + 3];

                        float c_j_x = Owner.Centroids.Host[j * CENTROID_FIELDS];
                        float c_j_y = Owner.Centroids.Host[j * CENTROID_FIELDS + 1];
                        float c_j_vx = Owner.Centroids.Host[j * CENTROID_FIELDS + 2];
                        float c_j_vy = Owner.Centroids.Host[j * CENTROID_FIELDS + 3];

                        bool intersects = (Math.Abs(c_i_x - c_j_x) * 0.5f < c_i_vx + c_j_vx) && (Math.Abs(c_i_y - c_j_y) * 0.5f < c_i_vy + c_j_vy);

                        if (intersects)
                        {
                            m_1meansKernel.Run(Owner.AttentionMap, i, j, Owner.m_centroidsCount + cIndex, Owner.ReductionSources, m_imageWidth, m_imageHeight);

                            for (int k = 0; k < Owner.m_sumsPerItem; k++)
                            {
                                //m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count,
                                //    (Owner.m_centroidsCount + cIndex) * Owner.m_sumsPerItem + k,
                                //    (Owner.m_centroidsCount + cIndex) * Owner.AttentionMap.Count * Owner.m_sumsPerItem + k * Owner.AttentionMap.Count, 1, /* distributed: */ 0);
                                m_reduction_kernel.size = Owner.AttentionMap.Count;
                                m_reduction_kernel.outOffset = (Owner.m_centroidsCount + cIndex) * Owner.m_sumsPerItem + k;
                                m_reduction_kernel.inOffset = (Owner.m_centroidsCount + cIndex) * Owner.AttentionMap.Count * Owner.m_sumsPerItem + k * Owner.AttentionMap.Count;
                                m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);
                            }
                        }
                        cIndex++;
                    }
                }

                //..and finalize their stats
                int combiCount = (Owner.m_centroidsCount - 1) * Owner.m_centroidsCount / 2;

                m_finalize_kernel.SetupExecution(combiCount);
                m_finalize_kernel.Run(Owner.Centroids, combiCount, Owner.Statistics, LearningRate, Owner.m_centroidsCount);

                Owner.Centroids.SafeCopyToHost();
                Owner.CentroidsDBI.SafeCopyToHost();

                cIndex = 0;
                float maxGain = 0;
                int toDelete1 = -1;
                int toDelete2 = -1;
                int maxGainIndex = -1;

                //iterate through joins and find maximum gain
                for (int j = 0; j < Owner.m_centroidsCount; j++)
                {
                    for (int i = j + 1; i < Owner.m_centroidsCount; i++)
                    {
                        float dbi_n = Owner.Centroids.Host[(Owner.m_centroidsCount + cIndex) * CENTROID_FIELDS + 5];
                        float dbi_cij = Owner.CentroidsDBI.Host[j * MAX_CENTROIDS + i];

                        //it the join better then separation?
                        if (dbi_n < dbi_cij && dbi_cij - dbi_n > maxGain)
                        {
                            maxGain = dbi_cij - dbi_n;
                            maxGainIndex = cIndex;
                            toDelete1 = i;
                            toDelete2 = j;
                        }
                        cIndex++;
                    }
                }

                //apply join on the best candidate
                if (maxGainIndex != -1)
                {                    
                    //repace 1st to join with the best candidate 
                    Owner.Centroids.CopyToMemoryBlock(Owner.Centroids, (Owner.m_centroidsCount + maxGainIndex) * CENTROID_FIELDS, toDelete1 * CENTROID_FIELDS, CENTROID_FIELDS);
                    Owner.m_centroidsCount--;

                    //replace 2nd to join with the last centroid
                    if (toDelete2 != Owner.m_centroidsCount)
                    {
                        Owner.Centroids.CopyToMemoryBlock(Owner.Centroids, Owner.m_centroidsCount * CENTROID_FIELDS, toDelete2 * CENTROID_FIELDS, CENTROID_FIELDS);
                    }
                }
            }

            private void FocusToCentroid()
            {
                float currentFocusLevel = Owner.ControlValues.Host[0];
                int focusIndex = (int)Owner.ControlValues.Host[1];

                float sumAllWeights = Owner.Statistics.Host[Owner.m_centroidsCount * Owner.m_sumsPerItem];
                float sumImportanceDelta = 0;

                for (int i = 0; i < MAX_CENTROIDS - Owner.m_centroidsCount; i++)
                {
                    Owner.CentroidImportance.Host[Owner.m_centroidsCount + i] = 0;
                }

                //update centroids importance
                for (int i = 0; i < Owner.m_centroidsCount; i++)
                {
                    float importanceDelta = 0;

                    if (sumAllWeights > 0)
                    {
                        importanceDelta = Owner.Centroids.Host[CENTROID_FIELDS * i + 4] / sumAllWeights;
                    }
                    Owner.CentroidImportance.Host[i] += importanceDelta;

                    //suppress empty centroids (zero variance)
                    if (Owner.Centroids.Host[CENTROID_FIELDS * i + 2] == 0 || Owner.Centroids.Host[CENTROID_FIELDS * i + 3] == 0)
                    {
                        Owner.CentroidImportance.Host[i] = 0;
                    }
                    sumImportanceDelta += importanceDelta;
                }

                currentFocusLevel -= 1;

                //update current focus 
                if (currentFocusLevel <= 0)
                {
                    //select new centroid
                    float maxImportance = Owner.CentroidImportance.Host[0];
                    focusIndex = 0;

                    for (int i = 1; i < Owner.m_centroidsCount; i++)
                    {
                        if (Owner.CentroidImportance.Host[i] > maxImportance)
                        {
                            focusIndex = i;
                            maxImportance = Owner.CentroidImportance.Host[i];
                        }
                    }

                    Owner.CentroidImportance.Host[focusIndex] = 0;
                    currentFocusLevel = FocusInterval;
                }

                if (Owner.CentroidsCount > 0 && sumAllWeights > 0)
                {
                    //apply pupil control 
                    Owner.PupilControl.Host[0] = Owner.Centroids.Host[CENTROID_FIELDS * focusIndex];
                    Owner.PupilControl.Host[1] = Owner.Centroids.Host[CENTROID_FIELDS * focusIndex + 1];
                    Owner.PupilControl.Host[2] = Owner.Centroids.Host[CENTROID_FIELDS * focusIndex + 2] + Owner.Centroids.Host[CENTROID_FIELDS * focusIndex + 3] + ScaleBase;

                    Owner.ControlValues.Host[0] = currentFocusLevel;
                    Owner.ControlValues.Host[1] = focusIndex;
                    Owner.ControlValues.Host[2] = sumImportanceDelta;
                }
                else
                {
                    Owner.PupilControl.Host[0] = 0;
                    Owner.PupilControl.Host[1] = 0;
                    Owner.PupilControl.Host[2] = 0.5f;
                }
            }

            private void ExecuteSimple()
            {
                //evaluate distances to mean for all pixels
                m_kernel.Run(Owner.AttentionMap, Owner.ReductionSources, m_imageWidth, m_imageHeight);

                //sum all values needed for mean & stdDev
                //ZXC m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count, 0, 0, 1, /* distributed: */ 0);
                m_reduction_kernel.size = Owner.AttentionMap.Count;
                m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);

                //ZXC m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count, 1, Owner.AttentionMap.Count, 1, /* distributed: */ 0);
                m_reduction_kernel.size = Owner.AttentionMap.Count;
                m_reduction_kernel.outOffset = 1;
                m_reduction_kernel.inOffset = Owner.AttentionMap.Count;
                m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);

                //ZXC m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count, 2, 2 * Owner.AttentionMap.Count, 1, /* distributed: */ 0);
                m_reduction_kernel.size = Owner.AttentionMap.Count;
                m_reduction_kernel.outOffset = 2;
                m_reduction_kernel.inOffset = 2 * Owner.AttentionMap.Count;
                m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);

                //ZXC m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources, Owner.AttentionMap.Count, 3, 3 * Owner.AttentionMap.Count, 1, /* distributed: */ 0);
                m_reduction_kernel.size = Owner.AttentionMap.Count;
                m_reduction_kernel.outOffset = 3;
                m_reduction_kernel.inOffset = 3 * Owner.AttentionMap.Count;
                m_reduction_kernel.Run(Owner.Statistics, Owner.ReductionSources);

                //sum all weights (denominator)
                //ZXC m_reduction_kernel.Run(Owner.Statistics, Owner.AttentionMap, Owner.AttentionMap.Count, 4, 0, 1, /* distributed: */ 0);
                m_reduction_kernel.outOffset = 4;
                m_reduction_kernel.Run(Owner.Statistics, Owner.AttentionMap);


                //evaluate mean & stdDev
                m_finalize_kernel.Run(Owner.Statistics);

                //apply pupil control (position = mean, radius = stdDev)
                m_eye_kernel.Run(Owner.PupilControl, Owner.Statistics, MoveFactor, ScaleFactor, ScaleBase);
            }
        }
    }
}


/* Gaussian creation snippet
 
if (SimulationStep == 0)                
{
    float amplitude = 1;
    float sigma = 0.1f;

    for (int i = 0; i < Owner.IgnoreMask.ColumnHint; i++) 
    {
        for (int j = 0; j < Owner.IgnoreMask.ColumnHint; j++)
        {
            float x = (float)i / Owner.IgnoreMask.ColumnHint * 2 - 1;
            float y = (float)j / Owner.IgnoreMask.ColumnHint * 2 - 1;

            float gValue = amplitude * (float)Math.Exp(-(x * x + y * y) / (2 * sigma));
            Owner.IgnoreMask.Host[j * Owner.IgnoreMask.ColumnHint + i] = gValue;
        }
    }

    Owner.IgnoreMask.SafeCopyToDevice();
}
 */