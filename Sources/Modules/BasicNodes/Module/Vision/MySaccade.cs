using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers; // Because of the keyboard...
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
//---- observers
using GoodAI.Modules.Vision;
using ManagedCuda;
using ManagedCuda.VectorTypes;           // manual kernel sizes are needed
using OpenTK.Input;
using System;
using System.ComponentModel;
using YAXLib;


namespace GoodAI.Modules.Vision
{

    /// <author>GoodAI</author>
    /// <meta>jk</meta>
    /// <status> Working </status>
    /// <summary>
    ///   Select input patch/segment/object for Saccadic movement...
    /// </summary>
    /// <description>
    ///   The node process set of hypothezied objects. Score each of them how likely it can be selected and selects the one.
    ///   <h4> Input </h4>
    ///   <ul>
    ///     <li>Patches:   [nx3]   matrix of x,y,scale of N patches/superpixels/objects. (-1 are ignored)</li>
    ///     <li>Desc:      [nxdim] descriptor of each path/superpixel/objects.</li>
    ///     <li>Mask:      [axb]   image where each pixels value corresponds to the segmetn id of the pixel.</li>
    ///     <li>ForcedXYS: [3x1]   force the output.</li>
    ///   </ul> 
    ///   
    ///   <h4> Output </h4>
    ///     Postion and size of the selected object/patch.
    ///
    ///  <h4> Parameters </h4>
    ///   <ul>
    ///     <li> Time Term / IncreaseOnFocus: How much to skip the object that the node selected last time.</li>
    ///     <li> Time Term / DecreaseInTime: How long it will take to again look on the patch the node selected before.</li>
    ///     <li> Terms Weighting / RationSupportMovement: How much favor moving elements.</li>
    ///   </ul>
    ///     
    /// 
    /// </description>
    public class MySaccade : MyWorkingNode
    {

        //----------------------------------------------------------------------------
        // :: MEMORY BLOCKS ::
        [MyInputBlock(0)]
        public MyMemoryBlock<float> Patches {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> Desc {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> Mask { 
            get { return GetInput(2); }
        }

        [MyInputBlock(3)]
        public MyMemoryBlock<float> ForcedXYS {
            get { return GetInput(3); }
        }
  


        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output {
            get { return GetOutput(0); }
            set { SetOutput(0, value); } 
        }
        
        public int MaskCount { get; set; }//{ get { return Mask != null ? Mask.Count : 0; } }
        public int MaskCols  { get; set; }//{ get { return Mask != null ? Mask.ColumnHint : 0; } }
        public int MaskRows  { get; set; }//{ get { return Mask != null ? Mask.Count / Mask.ColumnHint : 0; } }
        
        public int PatchesNum { get; set; }//{ get { return Patches != null ? Patches.Count / Patches.ColumnHint : 0; } }
        public int PatchesDim { get; set; }//{ get { return Patches != null ? Patches.ColumnHint : 0; } }

        public int Real_NPatches;  // real number of ppatches as there can be some zeros... :)

        public int DescNum { get; set; }//{ get { return Desc != null ? Desc.Count / Desc.ColumnHint : 0; } }
        public int DescDim { get; set; }//{ get { return Desc != null ? Desc.ColumnHint : 0; } }


        
        
        

        //----------------------------------------------------------------------------
        // :: DATA VARS  ::
        public MyMemoryBlock<float> EnergyTerms { get; private set; }
        public MyMemoryBlock<float> EnergyTotal { get; private set; }



        float Empty_energy_value = 1.0f;


        //----------------------------------------------------------------------------
        // :: INITS  ::
        public override void UpdateMemoryBlocks()
        {
            MaskCount = Mask != null ? Mask.Count : 1;
            MaskCols  = Mask != null ? Mask.ColumnHint : 1;
            MaskRows  = Mask != null ? MaskCount / MaskCols : 1;
            PatchesDim = Patches != null ? Patches.ColumnHint : 1;
            PatchesNum = Patches != null ? Patches.Count / PatchesDim : 1;
            DescDim = Desc != null ? Desc.ColumnHint : 1;
            DescNum = Desc != null ? Desc.Count / DescDim : 1;

            int dim_sccd_patch = 3;
            EnergyTerms.Count = PatchesNum * dim_sccd_patch;
            EnergyTerms.ColumnHint = dim_sccd_patch;
            EnergyTotal.Count = PatchesNum;
            EnergyTotal.ColumnHint = 1;

            Output.Count                  = 3;
        }

        public override void Validate(MyValidator validator)
        {
            //base.Validate(validator); /// base checking 
            validator.AssertError(Patches != null, this, "No input available");
            validator.AssertError(Desc != null, this, "No input available");
            validator.AssertError(Mask != null, this, "No input available");

        }









        
        public MySaccadeInitTask Init { get; private set; }

        /// <summary>
        /// Set default values for enery terms. It is done only once.
        /// </summary>
        [MyTaskInfo(OneShot = true)]  [Description("Init")]
        public class MySaccadeInitTask : MyTask<MySaccade>       
        {
            //private MyCudaKernel m_kernel_set_val;

            public override void Init(int nGPU)  { }

            public override void Execute()
            {
                Owner.EnergyTerms.Fill(Owner.Empty_energy_value);
                Owner.Output.Fill(-1.0f);
            }
        }







        
        public MySaccadeUpdateTask Update { get; private set; }

        /// <summary>
        /// Update terms based on the actual inputs -> now we have id of min energy.
        ///   <h4> Paramters </h4>
        ///   <ul>
        ///     <li> IncreaseOnFocus:       Term increase when focuser selects it. Lower will stay on the object longer.</li>
        ///     <li> DecreaseInTime:        Ration for decresing time term in time. Higher will try to go back on the seen patch sooner.</li>
        ///     <li> RationSupportMovement: How much to prefer movement against time.</li>
        /// </ul>
        /// </summary>
        [Description("Update")]
        public class MySaccadeUpdateTask : MyTask<MySaccade> 
        {

            [MyBrowsable, Category("Time Term"), YAXSerializableField(DefaultValue = 0.1f), Description("Term increase when focuser selects it. Lower will stay on the object longer")]
            public float IncreaseOnFocus { get; set; }

            [MyBrowsable, Category("Time Term"), YAXSerializableField(DefaultValue = 1.05f), Description("Ration for decresing time term in time. Higher will try to go back on the seen patch sooner.")]
            public float DecreaseInTime { get; set; }

            [MyBrowsable, Category("Terms Weighting"), YAXSerializableField(DefaultValue = 0.95f), Description("How much to prefer movement against time.")]
            public float RationSupportMovement { get; set; }



            private MyCudaKernel m_kernel_get_min_energy;
            private MyCudaKernel m_kernel_comp_total_energy;

            private MyCudaKernel m_kernel_EnTermUpdate_move;
            private MyCudaKernel m_kernel_EnTermUpdate_time;
            private MyCudaKernel m_kernel_set_val_between_idx;

            private MyCudaKernel m_kernel_reweight_based_on_brains_focus;

            public override void Init(int nGPU)
            {
                m_kernel_EnTermUpdate_move = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Saccade", "UdpateEnergyTerm_movement");
                m_kernel_EnTermUpdate_move.SetupExecution(Owner.EnergyTerms.Count);

                m_kernel_EnTermUpdate_time = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Saccade", "UdpateEnergyTerm_time");
                m_kernel_EnTermUpdate_time.SetupExecution(Owner.EnergyTerms.Count);

                m_kernel_comp_total_energy = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Saccade", "ComputeTotalEnergy");
                m_kernel_comp_total_energy.GridDimensions  = new dim3(Owner.PatchesNum);
                m_kernel_comp_total_energy.BlockDimensions = new dim3(Owner.EnergyTerms.ColumnHint);

                m_kernel_get_min_energy = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "getIdxOfMinVal_naive");
                m_kernel_get_min_energy.SetupExecution(new dim3(1), new dim3(1));
                m_kernel_set_val_between_idx = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\VisionMath", "SetVauleInIdxMinMax");
                m_kernel_set_val_between_idx.SetupExecution(Owner.EnergyTerms.Count);

                m_kernel_reweight_based_on_brains_focus = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Saccade", "ReeeigthTotalTermsThatBrainDoesNotFocusOn");
                m_kernel_reweight_based_on_brains_focus.SetupExecution(Owner.EnergyTotal.Count);
            }

            public override void Execute()
            {
                bool useBrainsForcedPostion = Owner.ForcedXYS != null;

                //--- set real number of patches
                Owner.Real_NPatches = EstimateRealNumOfPatches();

                //--- We may want only the part of the input matrix, thus delete values of energies of objects that are unused
                m_kernel_set_val_between_idx.Run(Owner.EnergyTerms, Owner.Real_NPatches * Owner.EnergyTerms.ColumnHint, Owner.EnergyTerms.Count - 1, Owner.Empty_energy_value);

                //--- Update movement energy term for the movement
                m_kernel_EnTermUpdate_move.Run(Owner.EnergyTerms, Owner.EnergyTerms.ColumnHint, Owner.Real_NPatches, Owner.Desc, Owner.DescDim, 3); //. 3 is index that stores movement info

                //--- cumulate energy terms into single one number that will be later search for min
                m_kernel_comp_total_energy.Run(Owner.EnergyTerms, Owner.EnergyTotal, Owner.Real_NPatches, Owner.EnergyTerms.ColumnHint, RationSupportMovement);

                //--- set you wont search energy terms that are not in the brains's favour
                if (useBrainsForcedPostion)
                {
                    m_kernel_reweight_based_on_brains_focus.Run(Owner.EnergyTotal, Owner.Real_NPatches, Owner.Patches, Owner.PatchesDim, Owner.ForcedXYS);
                }


                //--- get minimal energy's id as output
                m_kernel_get_min_energy.Run(Owner.EnergyTotal, Owner.Output, Owner.Real_NPatches);


                //--- Update time-change energy term FOR the NEXT STEP, thus use id-output to update it :)
                m_kernel_EnTermUpdate_time.Run(Owner.EnergyTerms, Owner.EnergyTerms.ColumnHint, Owner.Real_NPatches, Owner.Output,
                    IncreaseOnFocus, DecreaseInTime);
            }

            /// <summary>
            /// Find where zeros start -> this estimates # of patches...
            /// </summary>
            /// <returns></returns>
            private int EstimateRealNumOfPatches()
            {
                Owner.Patches.SafeCopyToHost();
                for (int i=0 ; i<Owner.PatchesNum ; i++){
                    float patch_sum = 0;
                    for (int j = 0; j < Owner.PatchesDim; j++)
                    {
                        patch_sum += Owner.Patches.Host[i * Owner.PatchesDim + j];
                    }
                    if (patch_sum == .0f)
                    {
                        return i; // current index (first not patch) is # of pathces till last one :)
                    }
                }
                return Owner.PatchesNum;
            }



            //--- if output should be directly ForcedXYS
            private void SetFocuserOnTheClosestObejcts()
            {
                //--- acess data from here
                Owner.Patches.SafeCopyToHost();
                Owner.ForcedXYS.SafeCopyToHost();
                //--- set output to the id of the closest patch
                Owner.Output.SafeCopyToHost();
                Owner.Output.Host[0] = GetIdOfClosetPatch(Owner.Patches.Host, Owner.Real_NPatches, Owner.ForcedXYS.Host, Owner.PatchesDim);
                Owner.Output.SafeCopyToDevice();
            }
            private int GetIdOfClosetPatch(float[] patches, int n_patches, float[] data_xys, int dim)
            {
                float min_dist = float.MaxValue;
                int id_winner = -1;
                for (int i = 0; i < n_patches; i++)
                {
                    float dist = SqrDist_between_vecs(patches, i, data_xys, 0, dim);
                    if (dist < min_dist)
                    {
                        id_winner = i;
                        min_dist = dist;
                    }
                }
                return id_winner;
            }
            public float SqrDist_between_vecs(float[] v1, int v1_start, float[] v2, int v2_start, int dim)
            {
                float dist = 0;
                for (int i = 0; i < dim; i++){
                    float t = v1[v1_start + i] - v2[v2_start + i];
                    dist += t * t;
                }
                return dist;
            }
        }







        
        public MySaccadeConvertOutId2XYTask ConvertOutId2XY { get; private set; }

        /// <summary>
        /// Convert id into a xy+scale and set it as the output
        /// </summary>
        [Description("ConvertOutId2XY")]
        public class MySaccadeConvertOutId2XYTask : MyTask<MySaccade>
        {
            public override void Init(int nGPU) {}

            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = false), Description("Wheter to output center postion as [-1,+1] or as pixel position.")]
            public bool ReturnRelativeToCenter { get; set; }

            public override void Execute()
            {
                Owner.Patches.SafeCopyToHost();
                Owner.Output.SafeCopyToHost();
                int id_row = (int)Owner.Output.Host[0];
                if (id_row == -1)
                {
                    System.Console.WriteLine(" -W- Saccade: id_row=-1");
                    return;
                }
                for (int i = 0; i < Owner.PatchesDim; i++)
                {
                    float dt = Owner.Patches.Host[id_row * Owner.PatchesDim + i];
                    if (ReturnRelativeToCenter)
                    { // convert XY postionin matrix into a relative postion to he center weigheted by the object size
                        if (i == 0)
                        { // it is column
                            dt -= Owner.MaskCols / 2;
                            dt /= Owner.MaskCols / 2;
                        }
                        else if (i == 1)
                        { // it is row
                            dt -= Owner.MaskRows / 2;
                            dt /= Owner.MaskRows / 2;
                        }
                        else if (i == 2)
                        { // it is picture's scale :)
                            dt /= Math.Min(Owner.MaskRows, Owner.MaskCols) / 2;
                            if (dt > 1)
                            { // this would give error in the focuser...
                                dt = 1;
                            }
                        }
                    }
                    Owner.Output.Host[i] = dt;
                }
                Owner.Output.SafeCopyToDevice();
            }
        }


    }
}









namespace GoodAI.Modules.Observers
{
    public class MySaccadeObserver : MyNodeObserver<MySaccade>
    {
        MyCudaKernel m_kernel_fillImWhite;
        MyCudaKernel m_kernel_fillByEnergy;
        MyCudaKernel m_kernel_trackFocuser;
        private CudaDeviceVariable<float> EyeMovementPathData;
        
        public MySaccadeObserver()
        {
            m_kernel_fillImWhite  = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\JoinPatchesObs", "FillImWhite");
            m_kernel_fillByEnergy = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\SaccadeObs", "FillImByEnergy");
            m_kernel_trackFocuser = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\SaccadeObs", "FocuserTracker");
        }

        protected override void Execute()
        {
            m_kernel_fillImWhite.SetupExecution(Target.MaskCount);
            m_kernel_fillByEnergy.SetupExecution(Target.MaskCount);
            m_kernel_trackFocuser.SetupExecution(Target.MaskCount);

            m_kernel_fillImWhite.Run(VBODevicePointer, TextureWidth, TextureHeight);

            var state = Keyboard.GetState();
            if (state[Key.K])
                m_kernel_trackFocuser.Run(VBODevicePointer, Target.MaskCount, Target.MaskCols, EyeMovementPathData.DevicePointer, EyeMovementPathData.Size, Target.Output, Target.Patches, Target.PatchesDim);
            else
                m_kernel_fillByEnergy.Run(VBODevicePointer, Target.Mask, Target.MaskCount, Target.EnergyTotal, 2.0f);
        }



        protected override void Reset()
        {
            TextureWidth = Target.MaskCols;
            TextureHeight = Target.MaskCount / TextureWidth;

            int n_eye_examples2store = 500;
            EyeMovementPathData = new CudaDeviceVariable<float>(n_eye_examples2store*2 + 1);
            EyeMovementPathData.Memset(0);
        }



        public float getValMax_inColumn(MyMemoryBlock<float> dt, int column)
        {
            dt.SafeCopyToHost();
            float max = -1000.0f;
            for (int i = 0; i < dt.Count / dt.ColumnHint; i++)
            {
                int idx = i * dt.ColumnHint + column;
                if (dt.Host[idx] > max)
                    max = dt.Host[idx];
            }
            return max;
        }

    }
}

