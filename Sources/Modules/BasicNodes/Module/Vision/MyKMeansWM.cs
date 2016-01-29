using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers; // Because of the keyboard...
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
//---- observers
using GoodAI.Modules.Vision;
using ManagedCuda.BasicTypes;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.Vision
{
    /// <author>GoodAI</author>
    /// <meta>jk,mv</meta>
    /// <status> Working</status>
    /// <summary>
    ///    On-the-fly clustering of incoming data. The node expects descriptor of the patch and its position. Then, it assigned cluster id to the input and change the clusters.
    /// </summary>
    /// <description>
    ///    Clustering based on the position and desc of the element.
    ///    
    ///   <h4> Inputs:</h4>
    ///    <ul>
    ///     <li>ObjectDesc:  [1xdim]   Descriptor.</li>
    ///     <li>ObjectXY:    [1x3]     XYS.</li>
    ///     <li>WorldEvent:  [1]       Information such game over is used in that input.</li>
    ///     <li>Image:       [N]       Image, used only for the visualization.</li>
    ///    </ul>
    ///    
    ///   <h4> Outputs:</h4>
    ///     <ul>
    ///     <li>NOfObjects:     [1]                 number of objects/clusters in the dbse.</li>
    ///     <li>ClusCenters:    [MaxClusters x dim] descriptors of each cluster.</li>
    ///     <li>ClusCentersXY:  [MaxClusters x 3]   XYS positions of each cluster.</li>
    ///    </ul>
    ///    
    /// 
    /// <h4>Paramters</h4>
    ///  <ul> 
    ///    <li> MaxClusters: Maximum number of elements in the memory. </li>
    ///    <li> ThresholdDescSim: Inverse distance that is necessary to assign input data into the existing memory element.</li>
    ///    <li> UpdateClusterCentersOrdering: Will I update database?</li>
    ///    <li> WeightXY: how to prefer XY similarity vs. descriptor similarity.</li>
    ///    <li> MoveClusCenterValue: Value to update cluster center, it is previous + c*newElemen</li>
    ///   </ul>
    ///   
    ///
    ///   <h4> Observer:</h4>
    ///     It shows positions of clusters in the XY plane. If the the Image input is used, it overlays it on the image.
    ///    
    /// </description>
    public class MyKMeansWM : MyWorkingNode
    {

        //----------------------------------------------------------------------------
        // MEMORY BLOCKS
        [MyInputBlock(0)]
        public MyMemoryBlock<float> ObjectDesc
        {
            get { return GetInput(0); }
        }
        [MyInputBlock(1)]
        public MyMemoryBlock<float> ObjectXY
        {
            get { return GetInput(1); }
        }
        [MyInputBlock(2)]
        public MyMemoryBlock<float> WorldEvent
        {
            get { return GetInput(2); }
        }
        [MyInputBlock(3)]
        public MyMemoryBlock<float> Image
        {
            get { return GetInput(3); }
        }
     



        [MyOutputBlock(0)]
        public MyMemoryBlock<float> NOfObjects
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }
        [MyOutputBlock(1)]
        public MyMemoryBlock<float> ClusCenters
        {
            get { return GetOutput(1); }
            private set { SetOutput(1, value); }
        }       //. cluster centers
        [MyOutputBlock(2)]
        public MyMemoryBlock<float> ClusCentersXY
        {
            get { return GetOutput(2); }
            private set { SetOutput(2, value); }
        }       //. to keep track about the size of each cluser center



        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 90), Description("Maximum number of elements in the memory.")]
        public int MaxClusters { get; set; }          //. # of clusters
                                                       

        
        //----------------------------------------------------------------------------
        // :: DATA VARS  ::
        [MyPersistable]
        public MyMemoryBlock<float> ClusCentersSize     { get; private set; }       //. to keep track about the size of each cluser center
        [MyPersistable]
        public MyMemoryBlock<float> ClusCentersLastSeen { get; private set; }       //. to keep track about the size of each cluser center
        public MyMemoryBlock<float> TempVal             { get; private set; }

        [MyPersistable]                                                             
        public MyMemoryBlock<float> ClusterDistSym { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<float> ClusterDistPos { get; private set; }
        [MyPersistable]
        public MyMemoryBlock<float> ClusterDists  { get; private set; }

        public MyMemoryBlock<float> WorldLostLifeCounter { get; private set; }


        public MyMemoryBlock<float> ReSort_idxArrayAscend { get; private set; }
        public MyMemoryBlock<float> ReSort_TmpClusters    { get; private set; }
        public MyMemoryBlock<float> ReSort_TmpClustersXY  { get; private set; }  //. not necassary, abocve tmp memory block should be larg eneough :)



        public int DescCount { get; set; }
        




        //----------------------------------------------------------------------------
        // :: INITS  ::
        public override void UpdateMemoryBlocks()
        {
            DescCount = ObjectDesc != null ? ObjectDesc.Count : 1;

            ReSort_TmpClusters.ColumnHint = ClusCenters.ColumnHint    = DescCount;
            ReSort_TmpClusters.Count      = ClusCenters.Count         = ClusCenters.ColumnHint * MaxClusters;

            ClusCentersSize.Count  = MaxClusters;

            ReSort_TmpClustersXY.ColumnHint = ClusCentersXY.ColumnHint = ObjectXY!=null ? ObjectXY.Count : 1;
            ReSort_TmpClustersXY.Count      = ClusCentersXY.Count      = MaxClusters * ClusCentersXY.ColumnHint;

            TempVal.Count = 2;

            ClusterDistSym.ColumnHint = ClusCentersLastSeen.ColumnHint = ClusterDistPos.ColumnHint = ClusterDists.ColumnHint = ReSort_idxArrayAscend.ColumnHint = 1;
            ClusterDistSym.Count      = ClusCentersLastSeen.Count = ClusterDistPos.Count = ClusterDists.Count = ReSort_idxArrayAscend.Count                     = MaxClusters;

            NOfObjects.Count = WorldLostLifeCounter.Count = 1;
        }

        public override void Validate(MyValidator validator)
        {
            //base.Validate(validator);
            validator.AssertError(ObjectDesc != null, this, "ObjectDesc not given.");
            validator.AssertError(ObjectXY != null,   this, "ObjectXY not given.");
            validator.AssertInfo(WorldEvent != null,  this, "WorldEvent input not given. no problem :)");
            /*if (MyMovement != null)
                validator.AssertError(MyMovement.Count >= 2, this, "Postion of myself has to be at least XY!");*/
        }







        
        public MyKMeansWMInitTask Init { get; private set; }
        /// <summary>
        /// 
        /// </summary>
        [MyTaskInfo(OneShot = true), Description("Init")]
        public class MyKMeansWMInitTask : MyTask<MyKMeansWM>       
        {

            public override void Init(int nGPU) {}

            public override void Execute()
            {
                Owner.ClusCentersSize.Fill(.0f);
                Owner.ClusCenters.Fill(.0f);
                Owner.ClusCentersLastSeen.Fill(.0f);
                Owner.NOfObjects.Fill(.0f);
            }
        }
        






        public MyKMeansWMExecuteTask Execute { get; private set; }
        /// <summary>
        ///  Process input data
        /// </summary>
        [Description("Execute")]
        public class MyKMeansWMExecuteTask : MyTask<MyKMeansWM> 
        {
            private MyCudaKernel m_kernel_AddNewCCenter;
            private MyCudaKernel m_kernel_UpadteCC_desc;
            private MyCudaKernel m_kernel_UpdateCC_XY;
            private MyCudaKernel m_kernel_UpdateXY_basedOnTheBrainsMovement;

            private MyProductKernel<float> m_dotKernel;
            private MyCudaKernel m_mulKernel;
            private MyCudaKernel m_matMultpl;
            private MyReductionKernel<float> m_minIdxKernel;




            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = .6f), Description("Inverse distance that is necessary to assign input data into the existing memory element.")]
            public float ThresholdDescSim { get; set; }   //. if checking whether to create new cluster node...

            [MyBrowsable, Category("Params"),YAXSerializableField(DefaultValue = false), Description("Update the database?")]
            public bool UpdateClusterCentersOrdering { get; set; }

            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = .2f), Description("XY similarity vs. descriptor similarity weigthing.")]
            public float WeightXY { get; set; }

            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = .08f), Description("Value to update cluster center, it is previous + c*newElement")]
            public float MoveClusCenterValue { get; set; }


            int   NearestCC_id;  //.  id of cloest
            float NearestCC_dist; //. similarity to closest


          
            public override void Init(int nGPU)
            {
                m_kernel_AddNewCCenter = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\KMeansWM", "AddDataAsCC");
                m_kernel_AddNewCCenter.SetupExecution(Owner.DescCount);

                m_kernel_UpadteCC_desc = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\KMeansWM", "UpadateCC_Desc");
                m_kernel_UpadteCC_desc.SetupExecution(Owner.DescCount);

                m_kernel_UpdateCC_XY = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\KMeansWM", "UpdateCC_XY");
                m_kernel_UpdateCC_XY.SetupExecution(Owner.ObjectXY.Count);


                m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, ProductMode.f_DotProduct_f);
                m_mulKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
                m_mulKernel.SetupExecution(Owner.DescCount);

                m_matMultpl = MyKernelFactory.Instance.Kernel(Owner.GPU, @"Common\CombineVectorsKernel", "MatMultipl_naive");
                m_matMultpl.GridDimensions = new ManagedCuda.VectorTypes.dim3(1, Owner.DescCount);
                m_matMultpl.BlockDimensions = new ManagedCuda.VectorTypes.dim3(1, 1);

                m_minIdxKernel = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_MinIdx_ff);

                m_kernel_UpdateXY_basedOnTheBrainsMovement = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\KMeansWM", "ApplyBrainsMovement");
                m_kernel_UpdateCC_XY.SetupExecution(Owner.MaxClusters);
            }

            public override void Execute()
            {
                Owner.NOfObjects.SafeCopyToHost();
                //--- will I create new cluster center? (yes if it is similar enough and if I have space for new cluster :)
                NormalizeVector(Owner.ObjectDesc, Owner.DescCount);
                GetNearestVector();

                if (SimulationStep == 0 && NearestCC_id == -1) //. When loading data I dont have corect size in the zero step... How to do it better?
                    NearestCC_id = 0;

                if (NearestCC_id == -1)
                { //. THIS NEEDS TO BE FIXED, THIS IS JUST TO DONT FALL WHEN ERROR!!!
                    NearestCC_id = 5;
                    System.Console.WriteLine(" Error in MyKMeansWM.cs with assigning NearestCC_id to 5 in step="+SimulationStep+". Just leave it, so set this variable somehow and go..");
                }

                /*if (Owner.MyMovement != null)
                { //--- If I moved, updated XY before I will compare them with the current focuser output :)
                    m_kernel_UpdateXY_basedOnTheBrainsMovement.Run(Owner.ClusCentersXY, Owner.ClusCentersXY.ColumnHint, Owner.MyMovement, Owner.MyMovement.Count, Owner.MaxClusters); //. move that cluster center
                }*/

                if (UpdateClusterCentersOrdering)
                {
                    if ( Owner.NOfObjects.Host[0] + 1 > Owner.MaxClusters || NearestCC_dist < ThresholdDescSim)
                    { //--- move that cluster
                        m_kernel_UpadteCC_desc.Run(Owner.ClusCenters, Owner.DescCount, NearestCC_id, Owner.ObjectDesc, Owner.ClusCentersSize, MoveClusCenterValue); //. move that cluster center
                        NormalizeVector(Owner.ClusCenters, Owner.DescCount, NearestCC_id); //. normalize the result
                    }
                    else
                    {  //--- Result is id new cluster :)
                        m_kernel_AddNewCCenter.Run(Owner.ClusCenters, Owner.DescCount, (int)Owner.NOfObjects.Host[0], Owner.ObjectDesc, Owner.ClusCentersSize);
                        NearestCC_id = (int)Owner.NOfObjects.Host[0];          //. nearset is last one and I will increase the size
                        Owner.NOfObjects.Host[0]++;
                        Owner.NOfObjects.SafeCopyToDevice();                //. Just to be sure if I will do CopyToHost (I dont) it will have correct number...
                    }
                }

                //--- update cluster I just focused on
                m_kernel_UpdateCC_XY.Run(Owner.ClusCentersXY, NearestCC_id, Owner.ObjectXY, Owner.ObjectXY.Count);

                //--- update when I saw it last time
                Update_ClusCentersLastSeen(); // keep what was sen last time etc...
                                              
                //--- set actual # of objects into the output
                Owner.NOfObjects.SafeCopyToDevice();
            }
            //--------------------------------------------------------------------
            /// <summary>
            /// 
            /// </summary>
            /// <param name="Vec">input vector</param>
            /// <param name="dim">dimetiosn of the vec</param>
            /// <param name="id_start">vec start dispalcement</param>
            private void NormalizeVector(MyMemoryBlock<float> Vec, int dim ,int id_start=0){
                CUdeviceptr VecDevPtr = Vec.GetDevicePtr(0,id_start * dim);

                //ZXC m_dotKernel.Run(Owner.TempVal, 0, VecDevPtr, VecDevPtr, dim, /* distributed: */ 0);
                m_dotKernel.Run(Owner.TempVal, VecDevPtr, VecDevPtr, dim);
                Owner.TempVal.SafeCopyToHost();
                float length = (float)Math.Sqrt(Owner.TempVal.Host[0]);

                if (length != 0)
                {
                    m_mulKernel.Run(0f, 0f, 1.0f / length, 0f, VecDevPtr, VecDevPtr, dim);
                }
                else
                {
                    Vec.Fill(0);
                }
            }

            //--------------------------------------------------------------------
            private void GetNearestVector()
            {
                NearestCC_dist = float.PositiveInfinity;
                NearestCC_id = -1;
                if (Owner.NOfObjects.Host[0] < 1)
                    return;

                //--- shapes
                m_matMultpl.Run(Owner.ClusCenters, Owner.ObjectDesc, Owner.ClusterDistSym, Owner.ClusCenters.ColumnHint, 1, Owner.ClusterDistSym.Count);
                Owner.ClusterDistSym.SafeCopyToHost();

                //--- XY
                Owner.ObjectXY.SafeCopyToHost();
                Owner.ClusCentersXY.SafeCopyToHost();
                float x = Owner.ObjectXY.Host[0];
                float y = Owner.ObjectXY.Host[1];

                for (int i = 0; i < Owner.NOfObjects.Host[0]; ++i)
                {
                    double dist = Math.Sqrt(Math.Pow((x - Owner.ClusCentersXY.Host[i * Owner.ClusCentersXY.ColumnHint]), 2.0) 
                        + Math.Pow((y - Owner.ClusCentersXY.Host[Owner.ClusCentersXY.ColumnHint * i + 1]), 2.0));
                    Owner.ClusterDistPos.Host[i] = (float)dist;
                }
                //--- check if I lost a life, then I wont use XY for distance :-)
                float use_XY = 1;
                if (Owner.WorldEvent != null)
                {
                    Owner.WorldEvent.SafeCopyToHost();
                    Owner.WorldLostLifeCounter.SafeCopyToHost();
                    if (Owner.WorldEvent.Host[0] < 0)
                    {
                        Owner.WorldLostLifeCounter.Host[0] = 20;
                    }
                    if (Owner.WorldLostLifeCounter.Host[0] > 0)
                    {
                        use_XY = 0;
                    }
                    if (Owner.WorldLostLifeCounter.Host[0] > 0)
                    {
                        --Owner.WorldLostLifeCounter.Host[0];
                    }
                        
                }

                //--- compute distance
                for (int i = 0; i < Owner.NOfObjects.Host[0]; ++i)
                {
                    Owner.ClusterDists.Host[i] = (1 - Owner.ClusterDistSym.Host[i]) + use_XY*WeightXY*Owner.ClusterDistPos.Host[i];
                }

                //--- copy necassary results to GPU
                Owner.ClusterDists.SafeCopyToDevice();
                Owner.ClusterDistPos.SafeCopyToDevice();
                if (Owner.WorldEvent != null)
                    Owner.WorldLostLifeCounter.SafeCopyToDevice();

                //--- update vars I will use
                for (int i = 0; i < Owner.NOfObjects.Host[0]; ++i)
                {
                    if (Owner.ClusterDists.Host[i] < NearestCC_dist)
                    {
                        NearestCC_dist = Owner.ClusterDists.Host[i];
                        NearestCC_id = i;
                    }
                }
            }
            //--------------------------------------------------------------------
            private void Update_ClusCentersLastSeen()
            {
                Owner.ClusCentersLastSeen.SafeCopyToHost();
                //Owner.ClusCentersLastSeen.Host[NearestCC_id] += 1.0f;   //. be sure that you didnt set dont create new clusters and you load precomputed data!
                Owner.ClusCentersLastSeen.Host[NearestCC_id] = SimulationStep;
                /*for (int i = 0; i < Owner.NOfObjects.Host[0]; i++)
                {
                    Owner.ClusCentersLastSeen.Host[i] *= 0.9f;// (1 + 1 / Owner.NUsedClusters);
                }*/
                Owner.ClusCentersLastSeen.SafeCopyToDevice();
            }


        }










        public MyKMeansWMReSortOutputTask ReSortOutput { get; private set; }

        /// <summary>
        /// In each iteration step, resort databse as FILO. So, the element that just arrived will be always first, the element that arrived before will be second, etc.
        /// </summary>
        [Description("Re-sort output")]
        public class MyKMeansWMReSortOutputTask : MyTask<MyKMeansWM>
        {
            private MyCudaKernel m_k_copyAB;
            private MyCudaKernel m_k_copyAB_shuffleIDX;

            public override void Init(int nGPU)
            {
                m_k_copyAB            = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\KMeansWM", "Copy_A_to_B");
                m_k_copyAB_shuffleIDX = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\KMeansWM", "Copy_matA_to_matB_withShuffleIdx");
            }



            public override void Execute()
            {
                //--- sort vector of 0,1,2,3,4... based on the LastSeenCluster value
                Owner.ReSort_idxArrayAscend.SafeCopyToHost();
                for (int i = 0; i < Owner.MaxClusters; i++)
                {
                    Owner.ReSort_idxArrayAscend.Host[i] = i;// Owner.MaxClusters - i - 1;
                }
                Array.Sort(Owner.ClusCentersLastSeen.Host, Owner.ReSort_idxArrayAscend.Host);
                Owner.ReSort_idxArrayAscend.SafeCopyToDevice();

                //--- resort the cluster centers based on that :)
                Reshuffle_Mat(Owner.ClusCenters, Owner.ReSort_TmpClusters, Owner.ReSort_idxArrayAscend);
                Reshuffle_Mat(Owner.ClusCentersXY, Owner.ReSort_TmpClustersXY, Owner.ReSort_idxArrayAscend);
                Reshuffle_Mat(Owner.ClusCentersSize, Owner.ReSort_TmpClustersXY, Owner.ReSort_idxArrayAscend);
                Reshuffle_Mat(Owner.ClusCentersLastSeen, Owner.ReSort_TmpClustersXY, Owner.ReSort_idxArrayAscend);
            }

            //--------------------------------------------------------------------
            /// <summary>
            ///   Change the ordering of vectors in Mat-matrix based on the indexes in IdxOrdering
            /// </summary>
            private void Reshuffle_Mat(MyMemoryBlock<float> Mat, MyMemoryBlock<float> TmpMat, MyMemoryBlock<float> IdxOrdering)
            {
                m_k_copyAB_shuffleIDX.SetupExecution(Mat.Count);
                m_k_copyAB_shuffleIDX.Run(Mat, TmpMat, Mat.Count, Mat.ColumnHint, IdxOrdering, Owner.MaxClusters); //. move that cluster center

                m_k_copyAB.SetupExecution(Mat.Count);
                m_k_copyAB.Run(TmpMat, Mat, Mat.Count);
            }
        }






        
    }
}







namespace GoodAI.Modules.Observers
{
    public class MyKMeansWMObserver : MyNodeObserver<MyKMeansWM>
    {
        MyCudaKernel m_kernel_fillImActSate;
        MyCudaKernel m_kernel_downInTime;

       // MyCudaKernel m_ker;

        
        public MyKMeansWMObserver()
        {
            m_kernel_fillImActSate = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\KMeansWM", "FillImByActState");
            m_kernel_downInTime    = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\KMeansWM", "DownInTime");

            //m_ker = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\VisionObsFce", "FillVBOFromInputImage");
        }

        protected override void Execute()
        {

            if (Target.Image == null)
            {
                Target.NOfObjects.SafeCopyToHost();
                m_kernel_downInTime.SetupExecution(TextureWidth * TextureHeight);
                m_kernel_downInTime.Run(VBODevicePointer, TextureWidth * TextureHeight);

                m_kernel_fillImActSate.GridDimensions = new ManagedCuda.VectorTypes.dim3((int)Target.NOfObjects.Host[0]); // numbe rof clusters to print
                m_kernel_fillImActSate.BlockDimensions = new ManagedCuda.VectorTypes.dim3(16, 16); // size of square
                //   m_kernel_fillImActSate.Run(VBODevicePointer, TextureWidth, TextureWidth * TextureHeight, Target.ClusCentersXY, Target.ClusCentersXY.ColumnHint, Target.MaxClusters, 1, Target.ClusCentersSize, GetMaxFreqOfClusters());
                m_kernel_fillImActSate.Run(VBODevicePointer, TextureWidth, TextureWidth * TextureHeight, Target.ClusCentersXY, Target.ClusCentersXY.ColumnHint, Target.MaxClusters, 1, Target.ClusCentersLastSeen, 2.0f, IsXYInputInOneNorm());
            }
            if (Target.Image != null)
            {
                MyCudaKernel m_ker = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\VisionObsFce", "FillVBOFromInputImage");
                m_ker.SetupExecution(TextureWidth * TextureHeight);
                m_ker.Run(Target.Image, TextureWidth * TextureHeight, VBODevicePointer);

                m_ker = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\KMeansWM", "FocuserInputObserver");
                m_ker.SetupExecution(TextureWidth * TextureHeight);

                m_ker.Run(Target.Image, Target.ClusCentersXY.GetDevicePtr(Target, 4 * Target.ClusCentersXY.ColumnHint), TextureWidth, TextureHeight, VBODevicePointer, 0.0f,0);
                m_ker.Run(Target.Image, Target.ClusCentersXY.GetDevicePtr(Target, 3 * Target.ClusCentersXY.ColumnHint), TextureWidth, TextureHeight, VBODevicePointer, 0.1f,0);
                m_ker.Run(Target.Image, Target.ClusCentersXY.GetDevicePtr(Target, 2 * Target.ClusCentersXY.ColumnHint), TextureWidth, TextureHeight, VBODevicePointer, 0.2f,0);
                m_ker.Run(Target.Image, Target.ClusCentersXY.GetDevicePtr(Target, 1 * Target.ClusCentersXY.ColumnHint), TextureWidth, TextureHeight, VBODevicePointer, 0.3f,0);
                m_ker.Run(Target.Image, Target.ClusCentersXY.GetDevicePtr(Target, 0 * Target.ClusCentersXY.ColumnHint), TextureWidth, TextureHeight, VBODevicePointer, 0.4f,0);
            }
        }

        int IsXYInputInOneNorm()
        {
            Target.ClusCentersXY.SafeCopyToHost();
            Target.NOfObjects.SafeCopyToHost();
            for (int i = 0; i < Target.NOfObjects.Host[0]; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    if (Target.ClusCentersXY.Host[j+i*Target.ClusCentersXY.ColumnHint] > 1.2)
                        return 0;
                }
            }
            return 1;
        }

        private float GetMaxFreqOfClusters(){   // return how oftewn was seen most often seen cluster
            Target.ClusCentersSize.SafeCopyToHost();
            Target.NOfObjects.SafeCopyToHost();
            float maxFreq = -1111;
            for (int i = 0; i < Target.NOfObjects.Host[0]; i++)
                if (Target.ClusCentersSize.Host[i] > maxFreq)
                    maxFreq = Target.ClusCentersSize.Host[i];
            return maxFreq;
        }

        protected override void Reset()
        {
            if (Target.Image == null)
            {
                TextureWidth = 600;
                TextureHeight = 600;
            }
            else
            {
                TextureWidth = Target.Image.ColumnHint;
                TextureHeight = Target.Image.Count / Target.Image.ColumnHint;
            }

        }

    }
}
