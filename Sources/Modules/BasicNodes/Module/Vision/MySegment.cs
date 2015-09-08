using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Observers;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
//---- observers
using GoodAI.Modules.Vision;
using ManagedCuda;
using ManagedCuda.VectorTypes;           // manual kernel sizes are needed
using System;
using System.ComponentModel;
using YAXLib;
//using OpenTK.Input; // Because of the keyboard...



namespace GoodAI.Modules.Vision
{


    /// <author>GoodAI</author>
    /// <meta>jk</meta>
    /// <status> Working </status>
    /// <summary>
    ///   Segment image into a set of superpixels. The code is restriceted to square images and specific nSegments values.
    /// </summary>
    /// <description>
    ///   Takes image (RGB or just Black and White if G/B input channels are left free) as an input and runs SLIC algorithm [1] to find segments. The code is calling the GPU version [2] of the 
    ///   SLIC method [1].
    ///   
    ///   <h4> Input</h4>
    ///     Image. When it is gray, use only the first branch.
    ///   
    ///   <h4> Output</h4>
    ///   <ul>
    ///     <li> SP_xy:    Center of each superpixel (segment) [x,y,nPts]  (need #of pts). It has size nSeg x 3.</li>
    ///     <li> SP_desc:  Descriptor [r,g,b,movement] it has size nSeg x 4.</li>
    ///     <li> Mask:     Mask where the value of each pixel has segmentId. it has size of the input image.</li>
    ///   </ul>
    ///   
    ///   <h4> Parameters </h4>
    ///   <ul>
    ///     <li> nSegs:  Number of segmetns. Try to keep number such that int n exist: nSegs=n*n; Higher->faster.</li>
    ///     <li> Weight: Whether segmetns should prefer grid structure. 0.9 works the best for fishes, while 0.3 for phong.</li>
    ///     <li> nIters: Number of iterations. Something between 1-3 is usally more than enough.</li>
    ///   </ul>
    ///   
    ///   <h4>Restrictions</h4>
    ///   <ul>
    ///     <li> The width of the image has to correspond to its height.</li>
    ///     <li> If n is number of segments, there has to exist integer k such that k*k=n.</li>
    ///     <li> It is better to keep the number of segments per rows/columns nicely exactly divided by the number of pixels per row/column.</li>
    ///   </ul>
    ///   
    ///   <h4>Observer</h4>
    ///    When the observer is used to visualize the result, you can change Operation/ObserverMode to change what is shows:
    ///   <ul>
    ///     <li> Image with borders of segments</li>
    ///     <li> only borders of segmetns</li>
    ///     <li> XYZ space of colors</li>
    ///     <li> id of segments </li>
    ///     <li> id os segments normalized for beter view </li>
    ///     <li> center of each segment</li>
    ///  </ul>
    ///  
    /// <h4>References</h4>
    /// [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, PAMI vol. 34, num. 11, p. 2274 - 2282, May 2012.<br></br>
    /// [2] <a href="https://github.com/painnick/gSLIC"> gSLIC is an GPU implementation of Simple Iterative Linear Clustering (SLIC) superpixel segmentation algorithm. </a>. Online avelaible June, 2015
    /// 
    /// </description>
    public class MySegment : MyWorkingNode
    {

        //----------------------------------------------------------------------------
        // MEMORY BLOCKS
        [MyInputBlock(0)]    
        public MyMemoryBlock<float> InputR_BW 
        {
            get{ return GetInput(0); }
        }

        [MyInputBlock(1)]    
        public MyMemoryBlock<float> InputG 
        { 
            get{ return GetInput(1); } 
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> InputB 
        { 
            get { return GetInput(2); } 
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> SP_xy 
        { 
            get { return GetOutput(0); } 
            set { SetOutput(0, value); } 
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> SP_desc 
        { 
            get { return GetOutput(1); } 
            set { SetOutput(1, value); } 
        }

        [MyOutputBlock(2)]
        public MyMemoryBlock<float> Mask 
        { 
            get { return GetOutput(2); } 
            set { SetOutput(2, value); } 
        }

        public int InputSize { get; set; }//{ get { return InputR_BW != null ? InputR_BW.Count : 0; } }
        public int InputDimX { get; set; }//{ get { return InputR_BW != null ? InputR_BW.ColumnHint : 0; } } // height for SLIC
        public int InputDimY { get; set; }//{ get { return InputR_BW != null ? InputR_BW.Count / InputR_BW.ColumnHint : 0; } } // width for SLIC

        public bool Is_input_RGB = true;

        [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 256), Description("Number of segmetns. Try to keep number such that int n exist: nSegs=n*n; Higher->faster")]
        public int nSegs { get; set; }     // # of segemtns
                                                                                                                              
     
        

        //----------------------------------------------------------------------------
        // :: DATA VARS  ::
        [MyPersistable]
        public MyMemoryBlock<float4> floatBuffer { get; private set; }
        public MyMemoryBlock<int>    maskBuffer  { get; private set; }

        public MyMemoryBlock<float> features_tmp_edges { get; private set; }

        public struct MySLICClusterCenterObject {
            public float4 lab;
            public float2 xy;
            public int nPoints;
            public int x1, y1, x2, y2; // Why the hell does he use/define these paramters?
            public int dummy; // needed for alignment to 16 bytes, 48 bytes total
        }

        public MyMemoryBlock<MySLICClusterCenterObject> SLICClusterCenters { get; private set; }


   


        //----------------------------------------------------------------------------
        // :: INITS  ::
        public override void UpdateMemoryBlocks()
        {
            InputSize = InputR_BW != null ? InputR_BW.Count : 1;
            InputDimX = InputR_BW != null ? InputR_BW.ColumnHint : 1;   // height for SLIC
            InputDimY = InputR_BW != null ? InputSize / InputDimX : 1;  // width for SLIC
            
            floatBuffer.Count = InputSize;
            maskBuffer.Count  = InputSize;

            SLICClusterCenters.Count = nSegs; 

            SP_xy.ColumnHint = 3;     // x,y,size
            SP_xy.Count = nSegs * SP_xy.ColumnHint;

            SP_desc.ColumnHint = 4; // color in XYZ space + movement
            SP_desc.Count = nSegs * SP_desc.ColumnHint;
            
            features_tmp_edges.Count = InputSize;

            Mask.Count      = InputSize;
            Mask.ColumnHint = InputDimX;
        }

        public int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

        public override void Validate(MyValidator validator)
        {
           // base.Validate(validator); // base checking 
            //System.Console.WriteLine("-I- Seg:: use SLIC: Input image; Output1-xyNpoints; Output-desc; Output3-mask with ids");
            validator.AssertError(InputR_BW != null, this, "At leaast the Red - channel must be connected, then it will be Balck white image!");
            validator.AssertError(nSegs > 0, this, "The numbner of segments (nSegs) has to be >0!");
            validator.AssertError(  ((int)Math.Sqrt(nSegs)) * ((int)Math.Sqrt(nSegs)) == nSegs, this, "The number of segments has to be a*a=nSegs!!!");
            validator.AssertError(InputDimX == InputDimY , this, "Input image has to be square.");

            
            //--- check if I can nicely devide image into blocks -> than I can have the exact number of segmetns!!!
            int nClusterSize = (int)Math.Sqrt((float)iDivUp(InputDimX * InputDimY, nSegs));
            int nClustersPerCol = (int)iDivUp(InputDimX, nClusterSize); // original for arbitrary sizes
            int nClustersPerRow = (int)iDivUp(InputDimY, nClusterSize);
            validator.AssertWarning(  (nClustersPerCol * nClustersPerCol) == nSegs , this, "Be sure that sqrt(nSegs)/ImageWidth is integer. This can be a reason for later errors!");
        }









        
        public MySLICTask DoSLIC { get; private set; }

        /// <summary>
        /// Run SLIC algorithm that calculates segments.
        /// </summary>
        [Description("SLIC")]
        public class MySLICTask : MyTask<MySegment>
        {

            

            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 0.3f), Description("Whether segmetns should prefer grid structure. 0.9 works the best for fishes, while 0.3 for phong.")]
            public float Weight { get; set; }  // How to prefer grid result

            [MyBrowsable, Category("Params"), YAXSerializableField(DefaultValue = 2), Description("Number of iterations. Something between 1-3 is usally more than enough.")]
            public float nIters { get; set; }  // # of iterations
            


            private MyCudaKernel m_kernel_init;
            private MyCudaKernel m_kernel_iterKMeans;
            private MyCudaKernel m_kernel_updateCC;
            private MyCudaKernel m_kernel_kBw2XYZ;
            private MyCudaKernel m_kernel_kRgb2XYZ;
            private MyCudaKernel m_kernel_kRgb2LAB;

            private MyCudaKernel m_kernel_copy2OutputMat;

            
            int MAX_BLOCK_SIZE = 512; // this needs to correpond to the same var in the kernel!

            int nWidth, nHeight, nClustersPerRow, nClustersPerCol;
            int nSeg;

            public override void Init(int nGPU)
            {
                //--------------- INIT VARS
                Owner.Is_input_RGB = (Owner.InputB != null);
                //System.Console.WriteLine(" --- ::MySegNode:: Is RGB input = " + Owner.Is_input_RGB.ToString());

                //--- init pars for buffer init :)
                nWidth  = Owner.InputDimX;   // ?what does it do?
                nHeight = Owner.InputDimY;

                int nClusterSize        = (int)Math.Sqrt((float)Owner.iDivUp(nWidth * nHeight, Owner.nSegs));

                nClustersPerCol = (int)Owner.iDivUp(nHeight, nClusterSize); // original for arbitrary sizes
                nClustersPerRow = (int)Owner.iDivUp(nWidth, nClusterSize);
                int nBlocksPerCluster   = Owner.iDivUp(nClusterSize * nClusterSize, MAX_BLOCK_SIZE);

                nSeg =  nClustersPerCol* nClustersPerRow; // This has to be equal to Owner.nSegs, diff. dimension otherwise!
                                                          // 
               // System.Console.WriteLine(nSeg + " " + nClustersPerCol + " " + nClustersPerRow + " " + nClustersPerRow * nClustersPerCol + ", " + nWidth + " "+ nHeight);

                int nBlockWidth  = nClusterSize;
                int nBlockHeight = Owner.iDivUp(nClusterSize, nBlocksPerCluster);

                //--- finally, init kernels for SLIC...
                m_kernel_kBw2XYZ  = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "kBw2XYZ");
                m_kernel_kBw2XYZ.SetupExecution(Owner.InputR_BW.Count);
                m_kernel_kRgb2XYZ = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "kRgb2XYZ");
                m_kernel_kRgb2XYZ.SetupExecution(Owner.InputR_BW.Count);
                m_kernel_kRgb2LAB = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "kRgb2LAB");
                m_kernel_kRgb2LAB.SetupExecution(Owner.InputR_BW.Count);

                m_kernel_init = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "kInitClusterCenters");
                m_kernel_init.GridDimensions            = new dim3(nClustersPerCol);
                m_kernel_init.BlockDimensions           = new dim3(nClustersPerRow);

                m_kernel_iterKMeans = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "kIterateKmeans");
                m_kernel_iterKMeans.GridDimensions      = new dim3(nBlocksPerCluster, Owner.nSegs);
                m_kernel_iterKMeans.BlockDimensions     = new dim3(nBlockWidth, nBlockHeight);

                m_kernel_updateCC = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "kUpdateClusterCenters");
                m_kernel_updateCC.GridDimensions        = new dim3(nClustersPerCol);
                m_kernel_updateCC.BlockDimensions       = new dim3(nClustersPerRow);

                //--- 2copy maks to output...
                m_kernel_copy2OutputMat = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "Copy_intMat2Float");
                m_kernel_copy2OutputMat.SetupExecution(Owner.InputR_BW.Count);
            }

            public override void Execute()
            {
                //--- copy image to the fomart for SLIC, if RGB, use LAB format, if not, jsut from BW...
                if (Owner.Is_input_RGB)
                    m_kernel_kRgb2XYZ.Run(Owner.InputR_BW, Owner.InputG, Owner.InputB, Owner.floatBuffer, Owner.InputR_BW.Count);
                else
                    m_kernel_kBw2XYZ.Run(Owner.InputR_BW, Owner.floatBuffer, Owner.InputR_BW.Count);

                //--- init data to store centers of segments etc.                          
                m_kernel_init.Run(Owner.floatBuffer, Owner.InputDimX, Owner.InputDimY, Owner.SLICClusterCenters);
                
                //--- they found that 5 iterations have already given good result, I do one :)
                for (int i = 0; i < nIters; i++)
                {
                    m_kernel_iterKMeans.Run(Owner.maskBuffer, Owner.floatBuffer, nWidth, nHeight, Owner.nSegs, nClustersPerRow, Owner.SLICClusterCenters, nSeg, 1, Weight);
                    m_kernel_updateCC.Run(Owner.floatBuffer, Owner.maskBuffer, nWidth, nHeight, Owner.nSegs, Owner.SLICClusterCenters, nSeg);
                }
                m_kernel_iterKMeans.Run(Owner.maskBuffer, Owner.floatBuffer, nWidth, nHeight, Owner.nSegs, nClustersPerRow, Owner.SLICClusterCenters, nSeg, 1, Weight);

                m_kernel_copy2OutputMat.Run(Owner.Mask, Owner.maskBuffer, Owner.InputR_BW.Count);
                
            }
        }



        public MyCalcDescTask DoCalcDesc { get; private set; }

        /// <summary>
        /// Calculate desriptor - it has color desc and movement (differnece to previous result). Also commented code for soemthing with edges, but it is useless and should be done in a better way.
        /// </summary>
        [Description("Calc. Descriptors")]   
        public class MyCalcDescTask : MyTask<MySegment>
        {
            private MyCudaKernel m_kernel_desc;

            public override void Init(int nGPU)
            {
                m_kernel_desc = MyKernelFactory.Instance.Kernel(nGPU, @"Vision\Segment", "Desc");
                m_kernel_desc.SetupExecution(Owner.nSegs);
            }

            public override void Execute()
            {
                CudaDeviceVariable<MySLICClusterCenterObject> devSLICCCenter = Owner.SLICClusterCenters.GetDevice(Owner); // get pointer

                m_kernel_desc.Run(devSLICCCenter.DevicePointer, Owner.SP_xy, Owner.SP_desc, Owner.nSegs, Owner.SP_xy.ColumnHint, Owner.SP_desc.ColumnHint, Owner.InputDimX, Owner.InputDimY); // fill descriptor
            }
        }
    }
}



















namespace GoodAI.Modules.Observers
{
    public class MySegmentObserver : MyNodeObserver<MySegment>
    {
        MyCudaKernel m_kernel_draw;
        MyCudaKernel k_test;

        public enum MySegObsMode
        {
            ImSegBorders,
            SegBorders,
            XYZColors,
            SegmentId,
            SegmentIdNorm,
            SegmentCetners,
        }

        [MyBrowsable, Category("Operation"), YAXSerializableField(DefaultValue = MySegObsMode.ImSegBorders), Description("Visualization mode")]
        public MySegObsMode ObserverMode { get; set; }

        public MySegmentObserver()
        {
            m_kernel_draw = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\SegmentObs", "Draw");

            k_test = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\SegmentObs", "Test_draw_xy");
            k_test.SetupExecution(new dim3(1), new dim3(1));
        }


        protected override void Execute()
        {
            //var state = Keyboard.GetState();
            m_kernel_draw.SetupExecution(Target.InputSize);

            switch (ObserverMode)
            {
                case MySegObsMode.ImSegBorders:
                    MyCudaKernel m_ker = MyKernelFactory.Instance.Kernel(MyKernelFactory.Instance.DevCount - 1, @"Vision\VisionObsFce", "FillVBOFromInputImage");
                    m_ker.SetupExecution(Target.InputSize);
                    m_ker.Run(Target.InputR_BW, TextureWidth * TextureHeight, VBODevicePointer);
                    m_kernel_draw.Run(VBODevicePointer, Target.SLICClusterCenters.GetDevice(Target).DevicePointer, Target.maskBuffer, TextureWidth, TextureHeight, Target.nSegs, 4);
                    break;
                case MySegObsMode.SegBorders:
                    m_kernel_draw.Run(VBODevicePointer, Target.SLICClusterCenters.GetDevice(Target).DevicePointer, Target.maskBuffer, TextureWidth, TextureHeight, Target.nSegs, 0);
                    break;
                case MySegObsMode.XYZColors:
                    m_kernel_draw.Run(VBODevicePointer, Target.SLICClusterCenters.GetDevice(Target).DevicePointer, Target.maskBuffer, TextureWidth, TextureHeight, Target.nSegs, 1);
                    break;
                case MySegObsMode.SegmentId:
                    m_kernel_draw.Run(VBODevicePointer, Target.SLICClusterCenters.GetDevice(Target).DevicePointer, Target.maskBuffer, TextureWidth, TextureHeight, Target.nSegs, 2);
                    break;
                case MySegObsMode.SegmentIdNorm:
                    m_kernel_draw.Run(VBODevicePointer, Target.SLICClusterCenters.GetDevice(Target).DevicePointer, Target.maskBuffer, TextureWidth, TextureHeight, Target.nSegs, 3);
                    break;
                case MySegObsMode.SegmentCetners:
                    m_kernel_draw.Run(VBODevicePointer, Target.SLICClusterCenters.GetDevice(Target).DevicePointer, Target.maskBuffer, TextureWidth, TextureHeight, Target.nSegs, 3);
                    k_test.Run(VBODevicePointer, TextureWidth, Target.SP_xy, Target.SP_xy.ColumnHint, Target.nSegs);
                    break;
            }

        }


        protected override void Reset()
        {
            TextureWidth = Target.InputR_BW.ColumnHint;
            TextureHeight = Target.InputSize / TextureWidth;
        }

    }
}

