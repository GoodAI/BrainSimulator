using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    /// <author>GoodAI</author>
    /// <meta>df, rb</meta>
    /// <status>Working, binary symbols not implemented</status>
    /// <summary>Symbolic representataion of uniform square grid.
    /// Will be used for storing spatial data into symbols.</summary>
    /// <description>
    /// Parameters:<br />
    /// <ul>
    /// <li>X_POINTS - number of points (symbols) on the X-axis</li>
    /// <li>Y_POINTS - number of points (symbols) on the Y-axis</li>
    /// <li>SYMBOL_COUNT - overall number of symbols (X_POINTS + Y_POINTS), read only</li>
    /// <li>Mode - mode of the symbolic grid mode: Encode - encode x and y to the symbol, Decode - decode symbol to the x and y positions, Cleanup - clean symbol on the input</li>
    /// </ul>
    /// </description>
    public class MySymbolicGrid : MyRandomPool
    {
        
        [MyBrowsable, Category("Grid")]
        [YAXSerializableField(DefaultValue = 11)]
        public int X_POINTS { get; set; }

        [MyBrowsable, Category("Grid")]
        [YAXSerializableField(DefaultValue = 11)]
        public int Y_POINTS { get; set; }

        [MyBrowsable, Category("Grid")]
        [YAXSerializableField(DefaultValue = 22)]
        public int SYMBOL_COUNT { get; private set; }

        public enum MyGridMode
        {
            Encode,
            Decode,
            Cleanup
        }

        [MyBrowsable, Category("Grid")]
        [YAXSerializableField(DefaultValue = MyGridMode.Encode)]
        public MyGridMode Mode { get; set; }

        [MyInputBlock]
        public MyMemoryBlock<float> Input
        {
            get { return GetInput(0); }
        }

        public MyMemoryBlock<float> GridX { get; private set; }
        public MyMemoryBlock<float> GridY { get; private set; }

        public MyMemoryBlock<float> SymbolX { get; private set; }
        public MyMemoryBlock<float> SymbolY { get; private set; }

        public MyMemoryBlock<float> SymbolVector { get; private set; }
        public MyMemoryBlock<float> NormalizedInput { get; private set; }

        public MyMemoryBlock<float> Distance { get; private set; }

        // TASKS
        public MyInitGridTask InitGrid { get; private set; }
        public MySymbolizePositionTask SymbolizePosition { get; private set; }

        public override string Description
        {
            get
            {
                switch (Mode)
                {
                    case MyGridMode.Encode: return "position->symbol";
                    case MyGridMode.Decode: return "symbol->position";
                    case MyGridMode.Cleanup: return "symbol->clean(symbol)";
                    default: return "N/A";
                }
            }
        }

        protected override int PatternCount 
        { 
            get {return SYMBOL_COUNT; }
        
        }

        public override int Seed
        {
            get { return 2345; }
        }

        protected override string GlobalVariableName
        {
            get { return "SYMBOLIC_GRID_" + SymbolSize; }
        }

        public override void UpdateMemoryBlocks()
        {
            SYMBOL_COUNT = X_POINTS + Y_POINTS;
            if (Mode == MyGridMode.Decode)
            {
                Output.Count = 2;
                Output.ColumnHint = 1;

                NormalizedInput.Count = SymbolSize;
            }
            else if(Mode == MyGridMode.Encode)
            {
                Output.Count = SymbolSize;
                Output.ColumnHint = ColumnHint;

                NormalizedInput.Count = 2;
            }
            else if (Mode == MyGridMode.Cleanup)
            {
                Output.Count = SymbolSize;
                Output.ColumnHint = ColumnHint;

                NormalizedInput.Count = SymbolSize;
            }

            GridX.Count = X_POINTS;
            GridY.Count = Y_POINTS;

            SymbolVector.Count = PatternCount * SymbolSize;
            SymbolVector.ColumnHint = SymbolSize;

            SymbolX.Count = SymbolSize;
            SymbolY.Count = SymbolSize;

            Distance.Count = PatternCount;
        }


        protected override float[] GenerateRandomVectors()
        {
            return base.GenerateRandomVectors();
        }

        /// <summary>
        /// Initialization task for symbolic grid node<br />
        /// The space is discretized and symbols are generated
        /// </summary>
        [MyTaskInfo(OneShot = true), Description("Init symbolic grid")]
        public class MyInitGridTask : MyTask<MySymbolicGrid>
        {
            public override void Init(int nGPU)
            {
                
            }

            public override void Execute()
            {
                float xStep;
                float yStep;

                xStep = 1.00f / (float)(Owner.X_POINTS - 1);
                yStep = 1.00f / (float)(Owner.Y_POINTS - 1);

                for (int x = 0; x < Owner.X_POINTS; x++)
                {
                    Owner.GridX.Host[x] = x * xStep;
                }
                
                for (int y = 0; y < Owner.Y_POINTS; y++)
                {
                    Owner.GridY.Host[y] = y * xStep;
                }

                Owner.GridX.SafeCopyToDevice();
                Owner.GridY.SafeCopyToDevice();

                // generate symbols
                CudaDeviceVariable<float> codeVectors = MyMemoryManager.Instance.GetGlobalVariable<float>(
                    Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors);
                Owner.SymbolVector.GetDevice(Owner).CopyToDevice(codeVectors, 0, 0, Owner.SymbolVector.Count * sizeof(float));
            }
        }

        /// <summary>
        /// According to the Mode set in the node parameters, this tasks provides encoding, decoding or cleaning up the symbol<br />
        /// Parameters:<br />
        /// <ul>
        /// <li>X_MAX: maximum possible value on the x-axis (encodes interval  &lt;0,X_MAX &gt;)</li>
        /// <li>Y_MAX: maximum possible value on the y-axis (encodes interval  &lt;0,Y_MAX &gt;)</li>
        /// </ul>
        /// </summary>
        [Description("Symbolize position")]
        public class MySymbolizePositionTask : MyTask<MySymbolicGrid>
        {

            [MyBrowsable, Category("Input")]
            [YAXSerializableField(DefaultValue = 159.00f)]
            public float X_MAX { get; set; }

            [MyBrowsable, Category("Input")]
            [YAXSerializableField(DefaultValue = 139.00f)]
            public float Y_MAX { get; set; }

            private MyCudaKernel m_normalizePositionKernel;

            private MyCudaKernel m_interpolateSymbolsKernel;
            private MyCudaKernel m_sumSymbolsKernel;
            private MyCudaKernel m_sumBasicSymbolsKernel;

            private MyCudaKernel m_computeDistanceKernel;

            public override void Init(int nGPU)
            {
                m_normalizePositionKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SymbolicGrid", "NormalizePositionKernel");
                m_interpolateSymbolsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SymbolicGrid", "InterpolateSymbolsKernel");
                m_sumSymbolsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SymbolicGrid", "SumSymbolsKernel");
                m_sumBasicSymbolsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SymbolicGrid", "SumBasicSymbolsKernel");
                m_computeDistanceKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\SymbolicGrid", "ComputeDistanceKernel");
            }

            public override void Execute()
            {
                // ENCODE MODE
                if (Owner.Mode == MyGridMode.Encode)
                {
                    Owner.Input.SafeCopyToHost();

                    if (Owner.Input.Host[0] < 0)
                    {
                        MyLog.WARNING.WriteLine("MySymbolicGrid: incorrect X position: " + Owner.Input.Host[0]);
                        Owner.NormalizedInput.Host[0] = 0;
                    }
                    else
                    {
                        Owner.NormalizedInput.Host[0] = Owner.Input.Host[0] / X_MAX;
                    }

                    if (Owner.Input.Host[1] < 0)
                    {
                        MyLog.WARNING.WriteLine("MySymbolicGrid: incorrect Y position: " + Owner.Input.Host[1]);
                        Owner.NormalizedInput.Host[1] = 0;
                    }
                    else
                    {
                        Owner.NormalizedInput.Host[1] = Owner.Input.Host[1] / Y_MAX;
                    }
                    
                    float xStep = 1.00f / (float)(Owner.X_POINTS - 1);
                    float yStep = 1.00f / (float)(Owner.Y_POINTS - 1);

                    // compute indices on x-axis
                    int rightIdx;
                    int leftIdx = (int)Math.Floor(Owner.NormalizedInput.Host[0] / xStep);
                    if (leftIdx != Owner.X_POINTS - 1)
                    {
                        rightIdx = leftIdx + 1;
                    }
                    else
                    {
                        rightIdx = leftIdx;
                    }

                    float leftWeight = 1.00f - (Owner.NormalizedInput.Host[0] - Owner.GridX.Host[leftIdx]) / xStep;
                    float rightWeight = 1.00f - (Owner.GridX.Host[rightIdx] - Owner.NormalizedInput.Host[0]) / xStep;

                    // interpolate x vector
                    m_interpolateSymbolsKernel.SetupExecution(Owner.SymbolSize
                        );
                    m_interpolateSymbolsKernel.Run(Owner.SymbolVector,
                        leftIdx,
                        rightIdx,
                        leftWeight,
                        rightWeight,
                        Owner.SymbolX,
                        Owner.SymbolSize
                        );

                    // compute indices on y-axis
                    int downIdx;
                    int upIdx = (int)Math.Floor(Owner.NormalizedInput.Host[1] / yStep);
                    if (upIdx != Owner.Y_POINTS - 1)
                    {
                        downIdx = upIdx + 1;
                    }
                    else
                    {
                        downIdx = upIdx;
                    }

                    float upWeight = 1.00f - (Owner.NormalizedInput.Host[1] - Owner.GridY.Host[upIdx]) / yStep;
                    float downWeight = 1.00f - (Owner.GridY.Host[downIdx] - Owner.NormalizedInput.Host[1]) / yStep;

                    m_interpolateSymbolsKernel.SetupExecution(Owner.SymbolSize
                        );
                    m_interpolateSymbolsKernel.Run(Owner.SymbolVector,
                        Owner.X_POINTS + upIdx,
                        Owner.X_POINTS + downIdx,
                        upWeight,
                        downWeight,
                        Owner.SymbolY,
                        Owner.SymbolSize
                        );

                    m_sumSymbolsKernel.SetupExecution(Owner.SymbolSize
                        );
                    m_sumSymbolsKernel.Run(Owner.SymbolX,
                        Owner.SymbolY,
                        Owner.Output,
                        Owner.SymbolSize
                        );
                }
                else
                {
                    Owner.Distance.Fill(0.00f);
                    m_computeDistanceKernel.SetupExecution(Owner.SYMBOL_COUNT
                        );
                    m_computeDistanceKernel.Run(Owner.SymbolVector,
                        Owner.Input,
                        Owner.Distance,
                        Owner.SymbolSize,
                        Owner.SYMBOL_COUNT
                        );
                    Owner.Distance.SafeCopyToHost();

                    // DECODE MODE
                    if(Owner.Mode == MyGridMode.Decode)
                    {
                        float max = float.MinValue;
                        int maxXIdx = -1;
                        for (int i = 0; i < Owner.X_POINTS; i++)
                        {
                            if (Owner.Distance.Host[i] > max)
                            {
                                max = Owner.Distance.Host[i];
                                maxXIdx = i;
                            }
                        }

                        if (maxXIdx == -1)
                        {
                            HandleIncorrectPosition();
                            return;
                        }

                        int bestNeighborId = 0;
                        if (maxXIdx > 0 && maxXIdx < Owner.X_POINTS - 1)
                        {
                            if (Owner.Distance.Host[maxXIdx - 1] > Owner.Distance.Host[maxXIdx + 1])
                            {
                                bestNeighborId = maxXIdx - 1;
                            }
                            else
                            {
                                bestNeighborId = maxXIdx + 1;
                            }
                        }
                        else if(maxXIdx == 0)
                        {
                            bestNeighborId = 1;
                        }
                        else if (maxXIdx == Owner.X_POINTS - 1)
                        {
                            bestNeighborId = Owner.X_POINTS - 2;
                        }

                        float winnerWeight, bestNeighborWeight;
                        if (Owner.Distance.Host[bestNeighborId] < 0)
                        {
                            bestNeighborWeight = 0.00f;
                            winnerWeight = 1.00f;
                        }
                        else
                        {
                            winnerWeight = Owner.Distance.Host[maxXIdx] / (Owner.Distance.Host[maxXIdx] + Owner.Distance.Host[bestNeighborId]);
                            bestNeighborWeight = Owner.Distance.Host[bestNeighborId] / (Owner.Distance.Host[maxXIdx] + Owner.Distance.Host[bestNeighborId]);
                        }
                        Owner.Output.Host[0] = (winnerWeight * Owner.GridX.Host[maxXIdx] + bestNeighborWeight * Owner.GridX.Host[bestNeighborId]) * X_MAX;

                        max = float.MinValue;
                        int maxYIdx = -1;
                        for (int i = Owner.X_POINTS; i < Owner.X_POINTS + Owner.Y_POINTS; i++)
                        {
                            if (Owner.Distance.Host[i] > max)
                            {
                                max = Owner.Distance.Host[i];
                                maxYIdx = i - Owner.X_POINTS;
                            }
                        }

                        if (maxYIdx == -1)
                        {
                            HandleIncorrectPosition();
                            return;
                        }

                        bestNeighborId = 0;
                        if (maxYIdx > 0 && maxYIdx < Owner.Y_POINTS - 1)
                        {
                            if (Owner.Distance.Host[maxYIdx + Owner.X_POINTS - 1] > Owner.Distance.Host[maxYIdx + Owner.X_POINTS + 1])
                            {
                                bestNeighborId = maxYIdx - 1;
                            }
                            else
                            {
                                bestNeighborId = maxYIdx + 1;
                            }
                        }
                        else if (maxYIdx == 0)
                        {
                            bestNeighborId = 1;
                        }
                        else if (maxYIdx == Owner.Y_POINTS - 1)
                        {
                            bestNeighborId = Owner.Y_POINTS - 2;
                        }

                        if (Owner.Distance.Host[bestNeighborId + Owner.X_POINTS] < 0)
                        {
                            bestNeighborWeight = 0.00f;
                            winnerWeight = 1.00f;
                        }
                        else
                        {
                            winnerWeight = Owner.Distance.Host[maxYIdx + Owner.X_POINTS] / (Owner.Distance.Host[maxYIdx + Owner.X_POINTS] + Owner.Distance.Host[bestNeighborId + Owner.X_POINTS]);
                            bestNeighborWeight = Owner.Distance.Host[bestNeighborId + Owner.X_POINTS] / (Owner.Distance.Host[maxYIdx + Owner.X_POINTS] + Owner.Distance.Host[bestNeighborId + Owner.X_POINTS]);
                        }
                        Owner.Output.Host[1] = (winnerWeight * Owner.GridY.Host[maxYIdx] + bestNeighborWeight * Owner.GridY.Host[bestNeighborId]) * Y_MAX;
                        Owner.Output.SafeCopyToDevice();
                    }
                    // CLEAN UP MODE
                    else if(Owner.Mode == MyGridMode.Cleanup)
                    {
                        float max = float.MinValue;
                        int maxXIdx = -1;
                        for (int i = 0; i < Owner.X_POINTS; i++)
                        {
                            if (Owner.Distance.Host[i] > max)
                            {
                                max = Owner.Distance.Host[i];
                                maxXIdx = i;
                            }
                        }

                        max = float.MinValue;
                        int maxYIdx = -1;
                        for (int i = Owner.X_POINTS; i < Owner.SYMBOL_COUNT; i++)
                        {
                            if (Owner.Distance.Host[i] > max)
                            {
                                max = Owner.Distance.Host[i];
                                maxYIdx = i;
                            }
                        }

                        m_sumBasicSymbolsKernel.SetupExecution(Owner.SymbolSize
                            );
                        m_sumBasicSymbolsKernel.Run(Owner.SymbolVector,
                            maxXIdx,
                            maxYIdx,
                            Owner.Output,
                            Owner.SymbolSize
                            );
                    }
                }
            }

            private void HandleIncorrectPosition()
            {
                MyLog.WARNING.WriteLine("Incorrect position detected");
                Owner.Output.Fill(0);
            }
        }

        /*
        [MyBrowsable, Category("Grid")]
        [YAXSerializableField(DefaultValue = 10)]
        public int Steps { get; set; }

        public enum MyGridMode
        {
            Encode,
            Decode,
            Cleanup
        }

        [MyBrowsable, Category("Grid")]
        [YAXSerializableField(DefaultValue = MyGridMode.Encode)]
        public MyGridMode Mode { get; set; }

        public MyMemoryBlock<float> Grid { get; private set; }
        public MyMemoryBlock<float> Temp { get; private set; }
        public MyMemoryBlock<float> Dots { get; private set; }

        public override string Description
        {
            get
            {
                switch (Mode)
                {
                    case MyGridMode.Encode: return "position->symbol";
                    case MyGridMode.Decode: return "symbol->position";
                    case MyGridMode.Cleanup: return "symbol->clean(symbol)";
                    default: return "N/A";
                }
            }
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = SymbolSize;
            Output.ColumnHint = ColumnHint;
            Grid.Count = (4 * Steps + 2) * SymbolSize;
            Grid.ColumnHint = ColumnHint;

            Temp.Count = MyFourierBinder.GetTempBlockSize(SymbolSize);
            Dots.Count = (4 * Steps + 2);
            Dots.ColumnHint = 2 * Steps + 1;

            if (Mode == MyGridMode.Decode)
            {
                Output.Count = 2;
                Output.ColumnHint = 1;
            }
            else
            {
                Output.Count = SymbolSize;
                Output.ColumnHint = ColumnHint;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (Input != null)
            {
                switch (Mode)
                {
                    case MyGridMode.Encode:
                        {
                            validator.AssertError(Input.Count > 0, this, "Input must be greater then 0.");
                        }
                        break;
                    case MyGridMode.Decode:
                    case MyGridMode.Cleanup:
                        {
                            validator.AssertError(Input.Count == SymbolSize, this, "Input must be equal to symbol size.");
                        }
                        break;
                }
            }
        }

        public MyInitGridTask InitGrid { get; private set; }
        public MySymbolizePositionTask SymbolizePosition { get; private set; }
        
        [MyTaskInfo(OneShot = true), Description("Init symbolic grid")]
        public class MyInitGridTask : MyTask<MySymbolicGrid>
        {
            private MyFourierBinder m_binder;

            private MyCudaKernel m_dotKernel;
            private MyCudaKernel m_mulKernel;

            private MyCudaKernel getUnitaryVectorKernel;


            public override void Init(int nGPU)
            {
                m_binder = new MyFourierBinder(Owner, Owner.SymbolSize, Owner.Temp);

                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "AbsoluteValueKernel");
                m_kernel.SetupExecution(Owner.SymbolSize);

                m_mulKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Transforms\TransformKernels", "PolynomialFunctionKernel");
                m_mulKernel.m_kernel.SetupExecution(Owner.SymbolSize);

                m_dotKernel = MyReductionFactory.Kernel(nGPU, MyReductionFactory.Mode.f_DotProduct_f);

                //getUnitaryVectorKernel = MyKernelFactory.Instance.Kernel(Owner.GPU, " ");
            }

            private int GetSymbolOffset(MyCodeVector symbol)
            {
                return (int)symbol * Owner.SymbolSize * sizeof(float);
            }

            private int SymbolSizeBytes
            {
                get { return Owner.SymbolSize * sizeof(float); }
            }

            void Normalize(int symbolSize, params CUdeviceptr[] symbols)
            {
                if (symbols == null)
                    return;

                for (int i = 0; i < symbols.Length; i++)
                    m_dotKernel.Run(Owner.Temp, i, symbols[i], symbols[i], symbolSize, 0);

                Owner.Temp.SafeCopyToHost(0, symbols.Length);

                for (int i = 0; i < symbols.Length; i++)
                {
                    float length = (float)Math.Sqrt(Owner.Temp.Host[i]);

                    length = 1 / length;

                    //if (length > 0.000001f)
                    m_mulKernel.Run(0f, 0f, length, 0f, symbols[i], symbols[i], symbolSize);
                }

                //for (int i = 0; i < symbols.Length; i++)
                //{
                //    m_dotKernel.Run(Owner.Temp, i, symbols[i], symbols[i], symbolSize, 0);
                //    Owner.Temp.SafeCopyToHost(0, symbols.Length);
                //}
            }

            public override void Execute()
            {
                int yAxisOffset = 2 * Owner.Steps + 1;

                CudaDeviceVariable<float> codeVectors = MyMemoryManager.Instance.GetGlobalVariable<float>(
                    Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors);

                CUdeviceptr dirX = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.DirX);
                CUdeviceptr dirY = codeVectors.DevicePointer + GetSymbolOffset(MyCodeVector.DirY);

                CUdeviceptr originX = Owner.Grid.GetDevicePtr(Owner) + Owner.Steps * SymbolSizeBytes;
                CUdeviceptr originY = Owner.Grid.GetDevicePtr(Owner) + (yAxisOffset + Owner.Steps) * SymbolSizeBytes;

                //Copy origins to the center of grid
                Owner.Grid.GetDevice(Owner).CopyToDevice(codeVectors,
                    GetSymbolOffset(MyCodeVector.OriginX), Owner.Steps * SymbolSizeBytes, SymbolSizeBytes);

                Owner.Grid.GetDevice(Owner).CopyToDevice(codeVectors,
                    GetSymbolOffset(MyCodeVector.OriginY), (yAxisOffset + Owner.Steps) * SymbolSizeBytes, SymbolSizeBytes);

                //Generate symbols for both axis
                for (int i = 0; i < Owner.Steps; i++)
                {
                    //if (i > 0)
                    //{
                    //    m_binder.Denominator = Owner.SymbolSize * (float)Math.Sqrt(i + 1);
                    //}

                    //positive direction
                    m_binder.Bind(originX + i * SymbolSizeBytes, dirX, originX + (i + 1) * SymbolSizeBytes);
                    m_binder.Bind(originY + i * SymbolSizeBytes, dirY, originY + (i + 1) * SymbolSizeBytes);

                    //negative direction
                    m_binder.Unbind(originX - i * SymbolSizeBytes, dirX, originX - (i + 1) * SymbolSizeBytes);
                    m_binder.Unbind(originY - i * SymbolSizeBytes, dirY, originY - (i + 1) * SymbolSizeBytes);

                    // Normalize the vectors
                    Normalize(Owner.SymbolSize,
                        originX + (i + 1) * SymbolSizeBytes,
                        originY + (i + 1) * SymbolSizeBytes,
                        originX - (i + 1) * SymbolSizeBytes,
                        originY - (i + 1) * SymbolSizeBytes);
                }
            }
        }
        
        [Description("Symbolize position")]
        public class MySymbolizePositionTask : MyTask<MySymbolicGrid>
        {

            [MyBrowsable, Category("Grid")]
            [YAXSerializableField(DefaultValue = 1)]
            public float SpatialBound { get; set; }

            [MyBrowsable, Category("Cleanup")]
            [YAXSerializableField(DefaultValue = 0.2f)]
            public float CleanupThreshold { get; set; }

            private MyFourierBinder m_binder;
            private MyCudaKernel m_decodeKernel;

            public override void Init(int nGPU)
            {
                m_binder = new MyFourierBinder(Owner, Owner.SymbolSize, Owner.Temp);
                m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
                m_kernel.m_kernel.SetupExecution(Owner.SymbolSize);

                m_decodeKernel = MyKernelFactory.Instance.Kernel(nGPU, @"VSA\Mappers", "DecodeSignal");
                m_decodeKernel.m_kernel.SetupExecution(Owner.Steps * 4 + 2);
            }

            private int GetNumberOffset(int number, MyCodeVector axis)
            {
                int yAxisOffset = 2 * Owner.Steps + 1;

                if (axis == MyCodeVector.DirX)
                {
                    return (number + Owner.Steps) * Owner.SymbolSize * sizeof(float);
                }
                else if (axis == MyCodeVector.DirY)
                {
                    return (yAxisOffset + number + Owner.Steps) * Owner.SymbolSize * sizeof(float);
                }
                else
                {
                    throw new ArgumentException("No suitable axis provided");
                }
            }

            private int ConvertToSymbolIndex(float value)
            {
                int index = (int)Math.Round((value / SpatialBound) * Owner.Steps);

                index = Math.Min(index, Owner.Steps);
                index = Math.Max(index, -Owner.Steps);
                return index;
            }

            private float ConvertFromSymbolIndex(int index)
            {
                return (float)index * SpatialBound / Owner.Steps;
            }

            private int SymbolSizeBytes
            {
                get { return Owner.SymbolSize * sizeof(float); }
            }

            public override void Execute()
            {
                if (Owner.Mode == MyGridMode.Encode)
                {
                    Owner.Input.SafeCopyToHost();

                    int indexX = ConvertToSymbolIndex(Owner.Input.Host[0]);
                    int indexY = ConvertToSymbolIndex(Owner.Input.Host[1]);

                    
                    //Owner.Temp.GetDevice(Owner).CopyToDevice(Owner.Grid.GetDevice(Owner),
                      //  GetNumberOffset(indexX, MyCodeVector.DirX), 0, SymbolSizeBytes);

                    //Owner.Temp.GetDevice(Owner).CopyToDevice(Owner.Grid.GetDevice(Owner),
                      //  GetNumberOffset(indexY, MyCodeVector.DirY), SymbolSizeBytes, SymbolSizeBytes);
                    

                    m_kernel.Run(
                        Owner.Grid.GetDevicePtr(Owner) + GetNumberOffset(indexX, MyCodeVector.DirX),
                        Owner.Grid.GetDevicePtr(Owner) + GetNumberOffset(indexY, MyCodeVector.DirY),
                        Owner.Output, (int)MyJoin.MyJoinOperation.Addition, Owner.SymbolSize);
                }
                else
                {
                    int symbolsPerAxis = Owner.Steps * 2 + 1;

                    Owner.Dots.Fill(0);
                    m_decodeKernel.Run(Owner.Input, Owner.Grid, Owner.Dots, Owner.Steps * 4 + 2, Owner.SymbolSize);
                    Owner.Dots.SafeCopyToHost();

                    float maxX = -1, maxY = -1;
                    int maxXi = 0, maxYi = 0;

                    for (int i = 0; i < symbolsPerAxis; i++)
                    {
                        if (maxX < Owner.Dots.Host[i])
                        {
                            maxX = Owner.Dots.Host[i];
                            maxXi = i - Owner.Steps;
                        }
                        if (maxY < Owner.Dots.Host[i + symbolsPerAxis])
                        {
                            maxY = Owner.Dots.Host[i + symbolsPerAxis];
                            maxYi = i - Owner.Steps;
                        }
                    }

                    if (Owner.Mode == MyGridMode.Decode)
                    {
                        if (maxX < CleanupThreshold)
                        {
                            maxXi = 0;
                        }

                        if (maxY < CleanupThreshold)
                        {
                            maxYi = 0;
                        }

                        Owner.Output.Host[0] = ConvertFromSymbolIndex(maxXi);
                        Owner.Output.Host[1] = ConvertFromSymbolIndex(maxYi);

                        Owner.Output.SafeCopyToDevice();
                    }
                    else if (Owner.Mode == MyGridMode.Cleanup)
                    {
                        if (maxX > CleanupThreshold && maxY > CleanupThreshold)
                        {
                            m_kernel.Run(
                                Owner.Grid.GetDevicePtr(Owner) + GetNumberOffset(maxXi, MyCodeVector.DirX),
                                Owner.Grid.GetDevicePtr(Owner) + GetNumberOffset(maxYi, MyCodeVector.DirY),
                                Owner.Output, (int)MyJoin.MyJoinOperation.Addition, Owner.SymbolSize);
                        }
                        else if (maxX > CleanupThreshold)
                        {
                            Owner.Output.GetDevice(Owner).CopyToDevice(
                                Owner.Grid.GetDevicePtr(Owner) + GetNumberOffset(maxXi, MyCodeVector.DirX),
                                0, 0, SymbolSizeBytes);
                        }
                        else if (maxY > CleanupThreshold)
                        {
                            Owner.Output.GetDevice(Owner).CopyToDevice(
                                Owner.Grid.GetDevicePtr(Owner) + GetNumberOffset(maxYi, MyCodeVector.DirY),
                                0, 0, SymbolSizeBytes);
                        }
                        else
                        {
                            Owner.Output.Fill(0);
                        }
                    }
                }
            }
        }
        */
    }
}
