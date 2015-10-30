using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Observers.Helper;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
//---- observers
using GoodAI.Modules.Vision;
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;  // for PointF
using System.Linq;
using YAXLib;
//using OpenTK.Input; // Because of the keyboard...



namespace GoodAI.Modules.Vision
{
    /// <author>GoodAI</author>
    /// <meta>jk</meta>
    /// <status> Working </status>
    /// <summary>
    ///   Concatenate patches/segments into objects.
    /// </summary>
    /// <description>
    /// The node requires a set of patches/superpixels (each defined with its location x,y) as input and joins them into a groups.
    /// Within the every group, there have to exists a path from one input segment (S1) to the another one (S2) such that
    /// distance between each ngh. nodes on one way between S1 and S2 is smaller then the treshold.
    /// 
    ///   <h4> Inputs:</h4>
    ///    <ul>
    ///     <li>Patches: [nx3]   matrix of x,y,scale of n patches/superpixels/objects.</li>
    ///     <li>Desc:    [nxdim] descriptor of each path/superpixel/objects.</li>
    ///     <li>Mask:    [axb]   image where each pixels value corresponds to the segment id of the pixel.</li>
    ///    </ul>
    ///    
    ///   <h4> Outputs:</h4>
    ///    Same like input, but segmetns are concatenated. As the size of the output's Patches and Desc is smaller, undefined elements are with -1.
    ///    
    /// 
    /// 
    ///   <h4> Parameters </h4>
    ///   <ul>
    ///     <li> ThresholdDescSim:  Required similarity between vicinity descriptors within a group.</li>
    ///   </ul>
    /// 
    ///   <h4>Observer:</h4>
    ///    When the observer is used to visualize the result, you can change Operation/ObserverMode to change what is shows:
    ///   <ul>
    ///    <li> Mask. </li>
    ///    <li> Mask overlay with the graph.</li>
    ///    <li> Graph with highlighted weights. </li>
    ///    <li> Mask with ids. </li>
    ///   </ul>
    ///  
    /// </description>
    public class MyJoinPatches : MyWorkingNode
    {


        //------------------------------------------------------------------------------------------------------
        // MEMORY BLOCKS
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Patches
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Desc
        {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> Mask
        {
            get { return GetInput(2); }
        }


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> OutPatches
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> OutDesc
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> OutMask
        {
            get { return GetOutput(2); }
            set { SetOutput(2, value); }
        }

        [MyOutputBlock(3)]
        public MyMemoryBlock<float> Patches2Obj
        {
            get { return GetOutput(3); }
            set { SetOutput(3, value); }
        }

        [MyOutputBlock(4)]
        public MyMemoryBlock<float> AdjMatrix
        {
            get { return GetOutput(4); }
            set { SetOutput(4, value); }
        }




        public int MaskCols { get; set; }//{ get { return Mask != null ? Mask.ColumnHint : 0; } }
        public int MaskRows { get; set; }//{ get { return Mask != null ? Mask.Count / Mask.ColumnHint : 0; } }
        public int MaskCount { get; set; }//{ get { return Mask != null ? Mask.Count : 0; } }

        public int PatchesNum { get; set; }//{ get { return Patches != null ? Patches.Count / Patches.ColumnHint : 0; } }
        public int PatchesDim { get; private set; }//{ get { return Patches != null ? Patches.ColumnHint : 0; } }

        public int DescDim { get; private set; }//{ get { return Desc != null ? Desc.ColumnHint : 0; } }


        public bool Is_input_RGB = true;
        public int OF_desc_dim = 2; // dimension of NEW Optical flow descriptor







        //------------------------------------------------------------------------------------------------------
        // :: DATA VARS  ::
        public MyMemoryBlock<float> Mask_new { get; private set; }
        public MyMemoryBlock<float> CentersSum { get; private set; }





        //------------------------------------------------------------------------------------------------------
        // :: INITS  ::
        public override void UpdateMemoryBlocks()
        {
            MaskCount = Mask != null ? Mask.Count : 1;
            MaskCols = Mask != null ? Mask.ColumnHint : 1;
            MaskRows = Mask != null ? MaskCount / MaskCols : 1;
            PatchesDim = Patches != null ? Patches.ColumnHint : 1;
            PatchesNum = Patches != null ? Patches.Count / PatchesDim : 1;

            Mask_new.Count = MaskCount;
            Mask_new.ColumnHint = MaskCols;

            CentersSum.ColumnHint = 3;
            CentersSum.Count = PatchesNum * CentersSum.ColumnHint;


            DescDim = Desc != null ? Desc.ColumnHint : 1;

            OutPatches.Count = PatchesNum * PatchesDim; // I want to return patches again, they will be with changed ID :D
            OutPatches.ColumnHint = PatchesDim;
            Patches2Obj.Count = PatchesNum; // I want to return patches again, but some of them will shave same id :D
            Patches2Obj.ColumnHint = 1;
            OutMask.Count = MaskCount;
            OutMask.ColumnHint = MaskCols;
            AdjMatrix.Count = PatchesNum * PatchesNum;
            AdjMatrix.ColumnHint = PatchesNum;
            OutDesc.ColumnHint = DescDim;    // center + size + number of points inisde + # of patches insinde
            OutDesc.Count = PatchesNum * OutDesc.ColumnHint;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator); // base checking 
        }






        public MyProcessImPatchBasTask Execute { get; private set; }
        public MyCreateGraphTask CreateGraph { get; private set; }


        /// <summary>
        /// Execute joining patches into groups of objects.
        /// </summary>
        [Description("Execute")]
        public class MyProcessImPatchBasTask : MyTask<MyJoinPatches>
        {
            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 0.08f), Description("Required similarity between vicinity descriptors within a group.")]
            public float ThresholdDescSim { get; set; }

            private MyCudaKernel m_kernel;
            private MyCudaKernel m_kernel_resetIm;
            private MyCudaKernel m_kernel_cumulatePosOfNewpatches;

            //----  dynamic stuff for C#
            private int[] maskIds_new;
            private List<int> ListToGo;    // which nodes Im CURRENTLY going to visit
            private Dictionary<int, int> DictUnmet;   // whcih nodes were already viseted so I will never meet them again.

            private int[] nPatchesInObj; // = new int[Owner.PatchesNum];
            private float2[] centers;       // = new float2[Owner.PatchesNum];
            private int[] nPointsInObj;  //= new int[Owner.PatchesNum];


            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\JoinPatches", "FillAdjacencyMatrix");
                m_kernel.SetupExecution(Owner.MaskCount);

                m_kernel_resetIm = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "ResetImage");
                m_kernel_resetIm.SetupExecution(Owner.PatchesNum * Owner.PatchesNum);

                m_kernel_cumulatePosOfNewpatches = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\JoinPatches", "CumulatePositionOfNewObjects");
                m_kernel_cumulatePosOfNewpatches.SetupExecution(Owner.MaskCount);

                //---- init C# stuff
                maskIds_new = new int[Owner.PatchesNum];
                ListToGo = new List<int>();              // which nodes Im CURRENTLY going to visit
                DictUnmet = new Dictionary<int, int>();    // which nodes were already visited so I wont meet it again... THIS SHOULD BE TREE for large # of nodes!!!!

                nPatchesInObj = new int[Owner.PatchesNum];
                centers = new float2[Owner.PatchesNum];
                nPointsInObj = new int[Owner.PatchesNum];
            }



            public override void Execute()
            {
                //--- reset list
                for (int ip = 0; ip < Owner.PatchesNum; ip++)
                    maskIds_new[ip] = ip;

                //--- create adjency matrix
                m_kernel_resetIm.Run(Owner.AdjMatrix, Owner.PatchesNum * Owner.PatchesNum);
                m_kernel.Run(Owner.AdjMatrix, Owner.Mask, Owner.MaskCount, Owner.MaskCols, Owner.MaskRows, Owner.PatchesNum);

                //------------------- NOW CREATE EDGES AND FIDN CONN. COMPONENTS...
                Owner.OutDesc.Fill(.0f);
                Owner.OutPatches.Fill(.0f);
                Owner.Patches2Obj.Fill(.0f);
                Owner.Mask_new.Fill(.0f);
                Owner.CentersSum.Fill(.0f);

                Owner.Mask.SafeCopyToHost();
                Owner.Desc.SafeCopyToHost();
                Owner.Patches.SafeCopyToHost();
                Owner.AdjMatrix.SafeCopyToHost();
                Owner.OutPatches.SafeCopyToHost();
                Owner.Patches2Obj.SafeCopyToHost();
                Owner.OutMask.SafeCopyToHost();
                Owner.OutDesc.SafeCopyToHost();

                //--- find connected compenents -------------------
                //--- init  lists
                ListToGo.Clear();
                DictUnmet.Clear();
                for (int i = 0; i < Owner.PatchesNum; i++)
                {
                    DictUnmet.Add(i, i);
                    centers[i].x = .0f; // init data  that were before new...
                    centers[i].y = .0f;
                    nPointsInObj[i] = 0;
                    nPatchesInObj[i] = 0;
                }
                //--- DO PROPAGATION -----------------------------
                int maskId = 0;
                while (DictUnmet.Count > 0)
                { // find conn. compenents within the treshold for each new node
                    int patch_id = DictUnmet.First().Key;
                    ListToGo.Add(patch_id); // add to list for actual searching, the list has now only one element!
                    DictUnmet.Remove(patch_id);  //  I will check it now and never again :)
                    PropageSimToConcatenate(maskId++, ref ListToGo, ref DictUnmet); // 
                }

                //---- PROCESS COMPONENTS AND SAVE TO OUT ---------

                int nNew_patches = -1; // number of new patches = ,,objects''
                for (int ip = 0; ip < Owner.PatchesNum; ip++)
                {
                    //int id_obj = maskIds_new[ip];
                    int id_obj = (int)Owner.Mask_new.Host[ip];
                    nPatchesInObj[id_obj]++;
                    Owner.Patches2Obj.Host[ip] = id_obj;
                    for (int i = 0; i < Owner.Desc.ColumnHint; i++)
                    {
                        Owner.OutDesc.Host[id_obj * Owner.OutDesc.ColumnHint + i] += Owner.Desc.Host[ip * Owner.Desc.ColumnHint + i];
                    }
                    if (id_obj >= nNew_patches)
                    {
                        nNew_patches = id_obj;// maskIds_new[ip] + 1;
                    }
                }
                nNew_patches++; // it was oindex to the last object, not their count :)
                // 
                //--- update masks...
                Owner.Mask_new.SafeCopyToDevice();
                m_kernel_cumulatePosOfNewpatches.Run(Owner.Mask, Owner.Mask_new, Owner.OutMask, Owner.Mask.Count, Owner.Mask.ColumnHint, Owner.CentersSum, Owner.CentersSum.Count, Owner.CentersSum.ColumnHint);
                Owner.CentersSum.SafeCopyToHost();

                for (int ip = 0; ip < Owner.PatchesNum; ip++)
                { // create new patches structure of objects!!
                    int out_idDesc = ip * Owner.OutDesc.ColumnHint;
                    if (ip < nNew_patches)
                    { // if it is object
                        int out_idPatch = ip * Owner.OutPatches.ColumnHint;
                        float nPts = Owner.CentersSum.Host[2 + ip * Owner.CentersSum.ColumnHint];
                        Owner.OutPatches.Host[out_idPatch + 0] = Owner.CentersSum.Host[0 + ip * Owner.CentersSum.ColumnHint] / nPts;
                        Owner.OutPatches.Host[out_idPatch + 1] = Owner.CentersSum.Host[1 + ip * Owner.CentersSum.ColumnHint] / nPts;
                        Owner.OutPatches.Host[out_idPatch + 2] = (float)Math.Sqrt((double)nPts);
                        for (int i = 0; i < Owner.OutDesc.ColumnHint; i++)
                        { // there will be average over all super-pixels (!pathces no points)
                            Owner.OutDesc.Host[out_idDesc + i] /= nPatchesInObj[ip];
                        }

                    }
                    else
                    {
                        for (int i = 0; i < Owner.OutDesc.ColumnHint; i++)
                        {
                            Owner.OutDesc.Host[out_idDesc + i] = -1;
                        }
                    }
                }

                //---- copy to device
                Owner.OutPatches.SafeCopyToDevice();
                Owner.OutDesc.SafeCopyToDevice();
                Owner.Patches2Obj.SafeCopyToDevice();
            }



            //------------------------------------------------------------------------------------------------------
            // Functions that I need to run inside this Tasks...
            //------------------------------------------------------------------------------------------------------
            public float SqrDist_between_vecs(float[] v1, int v1_start, float[] v2, int v2_start, int dim)
            {
                float dist = 0;
                for (int i = 0; i < dim; i++)
                {
                    float t = v1[v1_start + i] - v2[v2_start + i];
                    dist += t * t;
                }
                return dist;
            }

            public void PropageSimToConcatenate(int maskId, ref List<int> ListToGo, ref Dictionary<int, int> DictUnmet)//,  List<int>[] edges)//List<KDTreeNodeList<int>> edges)
            {
                float[] adjMatrix = Owner.AdjMatrix.Host;
                int AdjMatCols = Owner.AdjMatrix.ColumnHint;

                MyMemoryBlock<float> desc = Owner.Desc;

                while (ListToGo.Count > 0)
                {
                    int id_patch = ListToGo[0];
                    Owner.Mask_new.Host[id_patch] = maskId;

                    //MyMemoryBlock<float> m = Owner.AdjMatrix;
                    ListToGo.RemoveAt(0);           // erase the current one; 
                    for (int id_ngh = 0; id_ngh < AdjMatCols; id_ngh++)
                    {
                        int id_host = id_ngh * AdjMatCols + id_patch;
                        float mat_value = adjMatrix[id_host];

                        if ((int)mat_value == 1)
                        { // check wheter it is neighbor
                            if (DictUnmet.ContainsKey(id_ngh))
                            { // it wasnt used
                                float desc_dist = SqrDist_between_vecs(desc.Host, id_patch * desc.ColumnHint, desc.Host, id_ngh * desc.ColumnHint, 1);//Owner.Desc.ColumnHint);
                                if (desc_dist < ThresholdDescSim)
                                { //  it is closer to the node and we need to relabel it, Juhuu :)  maskIds_new[id_ngh] > maskId
                                    ListToGo.Add(id_ngh);     // add to list to change val to current maskId
                                    DictUnmet.Remove(id_ngh); // mark that I will never visit it again...
                                    Owner.Mask_new.Host[id_patch] = maskId;
                                }
                            }
                        }
                    }
                }
            }

        }


        [Description("Create graph"), MyTaskInfo(Disabled = true, OneShot = false)]
        public class MyCreateGraphTask : MyTask<MyJoinPatches>
        {
            private MyCudaKernel m_kernel, m_kernel_resetIm;

            public override void Init(int nGPU)
            {
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\JoinPatches", "FillAdjacencyMatrix");
                m_kernel.SetupExecution(Owner.MaskCount);

                m_kernel_resetIm = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "ResetImage");
                m_kernel_resetIm.SetupExecution(Owner.PatchesNum * Owner.PatchesNum);
            }



            public override void Execute()
            {
                m_kernel_resetIm.Run(Owner.AdjMatrix, Owner.PatchesNum * Owner.PatchesNum);
                m_kernel.Run(Owner.AdjMatrix, Owner.Mask, Owner.MaskCount, Owner.MaskCols, Owner.MaskRows, Owner.PatchesNum);
            }
        }



    }
}






















namespace GoodAI.Modules.Observers
{
    public class MyJoinPatchesObserver : MyNodeObserver<MyJoinPatches>
    {
        MyCudaKernel m_kernel_drawEdges, m_kernel_fillImWhite, m_kernel_fillImFromIm, m_kernel_drawDesc;
        private CudaDeviceVariable<float> m_StringDeviceBuffer;

        public enum MyJoinPatObsMode
        {
            Mask,
            Graph,
            GraphWeights,
            MaskId,
            Desc
        }

        [MyBrowsable, Category("Operation"), YAXSerializableField(DefaultValue = MyJoinPatObsMode.Mask)]
        public MyJoinPatObsMode ObserverMode { get; set; }

        public MyJoinPatchesObserver()
        {
            m_kernel_fillImWhite = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\JoinPatchesObs", "FillImWhite");
            m_kernel_fillImFromIm = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\JoinPatchesObs", "FillImByOtherIm");
            m_kernel_drawEdges = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\JoinPatchesObs", "Draw_edges");
            m_kernel_drawDesc = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\JoinPatchesObs", "FillImByEnergy");

        }

        protected override void Execute()
        {
            m_kernel_drawEdges.GridDimensions = new dim3(Target.PatchesNum);
            m_kernel_drawEdges.BlockDimensions = new dim3(Target.PatchesNum);
            m_kernel_fillImWhite.SetupExecution(Target.MaskCount);
            m_kernel_fillImFromIm.SetupExecution(Target.MaskCount);

            m_kernel_fillImFromIm.Run(VBODevicePointer, Target.OutMask, TextureWidth * TextureHeight, 5);//(int)Target.PatchesNum /5);



            switch (ObserverMode)
            {
                case MyJoinPatObsMode.Graph:
                    //. print edges
                    m_kernel_drawEdges.Run(VBODevicePointer, TextureWidth, Target.AdjMatrix, Target.Patches, Target.PatchesNum, Target.PatchesDim, Target.Desc, Target.Desc.ColumnHint, 0);
                    break;
                case MyJoinPatObsMode.GraphWeights:
                    //. print weight of graph's edges
                    m_kernel_fillImWhite.Run(VBODevicePointer, TextureWidth, TextureHeight);
                    m_kernel_drawEdges.Run(VBODevicePointer, TextureWidth, Target.AdjMatrix, Target.Patches, Target.PatchesNum, Target.PatchesDim, Target.Desc, Target.Desc.ColumnHint, 1);
                    break;
                case MyJoinPatObsMode.MaskId:
                    //. print ids to objects
                    Target.OutPatches.SafeCopyToHost();
                    for (int i = 0; i < Math.Min(Target.PatchesNum, 10); i++)
                    {
                        int x = (int)Target.OutPatches.Host[i * Target.PatchesDim];
                        int y = (int)Target.OutPatches.Host[i * Target.PatchesDim + 1];
                        MyDrawStringHelper.String2Index(i.ToString(), m_StringDeviceBuffer);
                        MyDrawStringHelper.DrawStringFromGPUMem(m_StringDeviceBuffer, x, y, (uint)Color.White.ToArgb(), (uint)Color.Black.ToArgb(), VBODevicePointer, TextureWidth, TextureHeight,0,i.ToString().Length);
                    }
                    break;
                case MyJoinPatObsMode.Desc:
                    // find the max value
                    Target.Desc.SafeCopyToHost();
                    float maxValue = float.MinValue;
                    float minValue = float.MaxValue;
                    for (int i = 0; i < Target.Desc.Count; i++)
                    {
                        maxValue = (Target.Desc.Host[i] > maxValue) ? Target.Desc.Host[i] : maxValue;
                        minValue = (Target.Desc.Host[i] < minValue) ? Target.Desc.Host[i] : minValue;
                    }
                    maxValue = maxValue - minValue;
                    // draw first row in desc :)
                    m_kernel_drawDesc.SetupExecution(Target.MaskCount);
                    m_kernel_drawDesc.Run(VBODevicePointer, Target.Mask, TextureWidth * TextureHeight, Target.Desc, maxValue, minValue);
                    break;
            }


        }



        protected override void Reset()
        {
            m_StringDeviceBuffer = new CudaDeviceVariable<float>(1000);
            m_StringDeviceBuffer.Memset(0);
            TextureWidth = Target.Mask.ColumnHint;
            TextureHeight = Target.MaskCount / TextureWidth;
        }

    }
}

