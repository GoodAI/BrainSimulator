using System;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using ManagedCuda.BasicTypes;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    /// <author>GoodAI</author>
    /// <tag>#mm</tag>
    /// <status>Working</status>
    /// <summary>
    ///   Performs an element-wise join operation on the input vectors. This can be arithmetic (addition, multiplication,...), 
    ///   binary (AND, XOR,...), permutation, distance measurement or stacking the inputs.
    /// </summary>
    /// <description>This node is OBSOLETE and exists only because of the need for backward compatibility. It is to be removed in a future update.</description>
    public class MyJoin : MyWorkingNode, IMyVariableBranchViewNodeBase
    {
        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        private int OutputSize
        {
            get { return Output.Count; }
            set { Output.Count = value; }
        }

        [ReadOnly(false)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set
            {
                base.InputBranches = value;
                m_offsets = new int[value];
            }
        }

        [Obsolete("Use OutputDimensions please.")]
        [YAXSerializableField(DefaultValue = 0), YAXElementFor("IO")]
        private int OutputColHint { get; set; }

        [MyBrowsable, Category("I/O"), Description("Comma separated dimensions, such as \"2, 3, *\".")]
        [YAXSerializableField(DefaultValue = ""), YAXElementFor("IO")]
        public string OutputDimensions
        {
            get
            {
                // backward compatible layer: use OutputColHint saved in the project
                if (m_outputDimsHint.IsEmpty && OutputColHint > 0)
                {
                    m_outputDimsHint = new CustomDimensionsHint(OutputColHint);
                    OutputColHint = 0;  // Don't use the old value next time.
                }

                return m_outputDimsHint.PrintSource();
            }
            set
            {
                m_outputDimsHint = CustomDimensionsHint.Parse(value);
            }
        }

        private CustomDimensionsHint m_outputDimsHint;

        public int[] m_offsets = new int[0];

        public enum MyJoinOperation
        {
            /// DON'T CHANGE ORDERING OF THESE!!!!
            Addition,
            Subtraction,
            Multiplication,

            AND,
            OR,
            OR_thresholded,
            XOR,
            XNOR,
            IMP,

            Permutation,
            Inv_Permutation,

            Modulo,
            Division_int,

            Equal, // Warning: uses a strict equality comparison on floats
            /// DON'T CHANGE ORDERING OF THESE!!!!

            AddToIdcs,
            AddToIdcs_Normalize,
            GatherFromIdcs,

            DotProduct,
            CosineDistance,
            DistanceSquared,

            MatMultiplication, /// Matrix multiplication

            //must be last
            StackInputs
        }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = MyJoinOperation.Addition), YAXElementFor("Behavior")]
        public MyJoinOperation Operation { get; set; }

        public MyMemoryBlock<CUdeviceptr> InputBlocksPointers { get; private set; }
        public MyMemoryBlock<float> Temp { get; private set; }

        public MyInitTask InitMemoryMapping { get; private set; }
        public MyStackInputsTask StackInputs { get; private set; }

        public MyJoin()
        {
            InputBranches = 2;
        }


        //----- for init! ??? Honza, is there any differnet way??
        public int Input0Count { get { return GetInput(0) != null ? GetInput(0).Count : 0; } }
        public int Input0ColHint { get { return GetInput(0) != null ? GetInput(0).ColumnHint : 0; } }
        public int Input1Count { get { return GetInput(1) != null ? GetInput(1).Count : 1; } }
        public int Input1ColHint { get { return GetInput(1) != null ? GetInput(1).ColumnHint : 0; } }

        public override void UpdateMemoryBlocks()
        {
            int totalOutputs = 0;
            Output.ColumnHint = 1;

            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);

                if (ai == null)
                    continue;

                m_offsets[i] = totalOutputs;
                totalOutputs += ai.Count;

                if (Output.ColumnHint == 1 && ai.ColumnHint > 1)
                {
                    Output.ColumnHint = ai.ColumnHint;
                }
            }

            switch (Operation)
            {
                case MyJoinOperation.StackInputs:
                    OutputSize = totalOutputs;
                    break;

                case MyJoinOperation.AddToIdcs_Normalize:
                    Temp.Count = GetInputSize(1) + 1;
                    goto default;

                case MyJoinOperation.GatherFromIdcs:
                    OutputSize = GetInputSize(1);
                    break;

                case MyJoinOperation.DistanceSquared:
                case MyJoinOperation.CosineDistance:
                case MyJoinOperation.DotProduct:
                    OutputSize = 1;
                    Output.ColumnHint = 1;
                    InputBlocksPointers.Count = 2;
                    Temp.Count = GetInputSize(0);
                    break;

                case MyJoinOperation.MatMultiplication:
                    if (Input0ColHint == 0)
                        OutputSize = 0;
                    else
                        OutputSize = Input0Count / Input0ColHint * Input1ColHint; /// size of output matrix: #rows A  times #cols B
                    Output.ColumnHint = Input1ColHint;  /// # of columns in the output correspond to the # of columns in the first matrix
                    InputBlocksPointers.Count = 2;
                    break;

                default:
                    if (InputBranches > 2)
                    {
                        // All are validated to be of the same length
                        OutputSize = GetInputSize(0);
                        InputBlocksPointers.Count = InputBranches;
                    }
                    else // (if InputBranches == 2)
                    {
                        InputBlocksPointers.Count = InputBranches;

                        int max = 0;

                        for (int i = 0; i < InputBranches; i++)
                        {
                            var input = GetInput(i);

                            if (input != null && input.Count > max)
                                max = input.Count;
                        }

                        OutputSize = max;
                    }
                    break;
            }

            TensorDimensions adjustedDims;

            if(!m_outputDimsHint.TryToApply(Output.Dims, out adjustedDims) && !m_outputDimsHint.IsEmpty)
                MyLog.WARNING.WriteLine("Join node '{0}': Could not apply OutputDimensions.", Name);  // TODO(Premek): Be specific.

            Output.Dims = adjustedDims;  // Adjusted or original.
        }

        public override void Validate(MyValidator validator)
        {
            switch (Operation)
            {
                case MyJoinOperation.StackInputs:
                    return;

                case MyJoinOperation.DistanceSquared:
                case MyJoinOperation.CosineDistance:
                case MyJoinOperation.DotProduct:
                    validator.AssertError(InputBranches == 2, this, "Two operands are needed for distance measures");
                    break;

                case MyJoinOperation.MatMultiplication:
                    bool is_correct = Input0ColHint == (Input1Count / Input1ColHint);
                    // if (Input1ColHint==1) /// BrainSim. bug for Nx1 vector, column hint is one, although it should be N...
                    //     is_correct = Input0ColHint == Input1Count;
                    validator.AssertError(is_correct, this, "# of columns in Mat1 needs to correspond to # of rows in Mat2!");
                    break;

                case MyJoinOperation.AddToIdcs:
                case MyJoinOperation.AddToIdcs_Normalize:
                    validator.AssertError(InputBranches >= 3, this, "Must provide the Target vector, Source vector and the idcs as the first three inputs");
                    validator.AssertError(GetInputSize(1) == GetInputSize(2), this, "Dimensions of the Source vector and idcs must be the same");
                    validator.AssertError(GetInputSize(0) >= GetInputSize(1), this, "Target vector must be bigger than Source vector");
                    return;

                case MyJoinOperation.GatherFromIdcs:
                    validator.AssertError(InputBranches >= 2, this, "Must provide the Target vector and the idcs as the first two inputs");
                    validator.AssertError(GetInputSize(0) >= GetInputSize(1), this, "Target vector must be bigger than the idcs");
                    return;

                default:
                    validator.AssertError(InputBranches >= 2, this, "Operation needs at least two operands");
                    break;
            }

            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);

                if (ai == null)
                    validator.AddError(this, string.Format("Missing input {0}.", i));
                else if (InputBranches > 2) // Two inputs are allowed to be of variable size
                    validator.AssertError(ai.Count == OutputSize, this, "Operand size differs from output size");
            }
        }

        public override string Description
        {
            get
            {
                return Operation.ToString();
            }
        }


        /// <summary>
        ///   Initializes any memory needed to perform the join operation.
        /// </summary>
        [Description("Init memory mapping"), MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<MyJoin>
        {
            public override void Init(int nGPU) { }

            public override void Execute()
            {
                switch (Owner.Operation)
                {
                    case MyJoinOperation.StackInputs:
                    case MyJoinOperation.GatherFromIdcs:
                        break;

                    default:
                        for (int i = 0; i < Owner.InputBranches; i++)
                        {
                            MyMemoryBlock<float> ai = Owner.GetInput(i);
                            Owner.InputBlocksPointers.Host[i] = ai != null ? ai.GetDevicePtr(Owner) : default(CUdeviceptr);
                        }

                        Owner.InputBlocksPointers.SafeCopyToDevice();
                        break;
                }
            }
        }

        /// <summary>
        ///   Performs the desired join operation.
        /// </summary>
        [Description("Perform join operation")]
        public class MyStackInputsTask : MyTask<MyJoin>
        {
            MyMemoryBlock<float> in0, in1, out0;

            private MyCudaKernel m_kernel;
            private MyProductKernel<float> m_dotKernel;
            private MyCudaKernel m_mapToIdcsKernel;

            public override void Init(int nGPU)
            {
                in0 = Owner.GetInput(0);
                in1 = Owner.GetInput(1);
                out0 = Owner.GetOutput(0);

                m_kernel = Owner.InputBranches > 2
                    ? MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel")
                    : MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernelVarSize");

                m_kernel.SetupExecution(out0.Count);

                switch (Owner.Operation)
                {
                    case MyJoinOperation.AddToIdcs:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "AddToIdcs");
                        m_kernel.SetupExecution(in1.Count);
                        break;

                    case MyJoinOperation.AddToIdcs_Normalize:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
                        m_kernel.SetupExecution(in1.Count);
                        m_mapToIdcsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "MapToIdcs");
                        m_mapToIdcsKernel.SetupExecution(in1.Count);
                        m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, ProductMode.f_DotProduct_f);
                        break;

                    case MyJoinOperation.GatherFromIdcs:
                        m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
                        m_kernel.SetupExecution(in1.Count);
                        break;

                    case MyJoinOperation.DotProduct:
                    case MyJoinOperation.DistanceSquared:
                        m_kernel.SetupExecution(in0.Count);
                        m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, ProductMode.f_DotProduct_f);
                        break;

                    case MyJoinOperation.CosineDistance:
                        m_dotKernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, ProductMode.f_Cosine_f);
                        break;

                    case MyJoinOperation.MatMultiplication:
                        {
                            // out0.Count / out0.ColumnHint
                            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "MatMultipl_naive");
                            int MAX_BLOCK_SIZE = 1;
                            m_kernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(out0.ColumnHint / MAX_BLOCK_SIZE, out0.Count / out0.ColumnHint / MAX_BLOCK_SIZE);
                            m_kernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);
                        }
                        break;
                }
            }

            public override void Execute()
            {
                switch (Owner.Operation)
                {
                    case MyJoinOperation.StackInputs:
                        for (int i = 0; i < Owner.InputBranches; i++)
                        {
                            MyMemoryBlock<float> ai = Owner.GetInput(i);
                            if (ai != null)
                            {
                                ai.CopyToMemoryBlock(out0, 0, Owner.m_offsets[i], ai.Count);
                            }
                        }
                        break;

                    case MyJoinOperation.AddToIdcs:
                        if (in0 != out0)
                            in0.CopyToMemoryBlock(out0, 0, 0, in0.Count);

                        m_kernel.Run(in1, Owner.GetInput(2), out0, (int)MyJoinOperation.Addition, in1.Count);
                        break;

                    case MyJoinOperation.AddToIdcs_Normalize:
                        var in2 = Owner.GetInput(2);
                        var temp = Owner.Temp;

                        if (in0 != out0)
                            in0.CopyToMemoryBlock(out0, 0, 0, in0.Count);

                        m_kernel.Run(in0, in2, temp, (int)MyJoinOperation.Permutation, in2.Count);
                        m_kernel.Run(in1, temp, temp, (int)MyJoinOperation.Addition, in1.Count);
                        //ZCX m_dotKernel.Run(temp, in1.Count, temp, temp, in1.Count, /* distributed: */ 0);
                        m_dotKernel.size = in1.Count;
                        m_dotKernel.outOffset = in1.Count;
                        m_dotKernel.Run(temp, temp, temp);
                        m_mapToIdcsKernel.Run(temp, temp.GetDevicePtr(Owner.GPU, in1.Count), in2, out0, in2.Count);
                        break;

                    case MyJoinOperation.GatherFromIdcs:
                        m_kernel.Run(in0, in1, out0, (int)MyJoinOperation.Permutation, in1.Count);
                        break;

                    case MyJoinOperation.DistanceSquared:
                        m_kernel.Run(in0, in1, Owner.Temp, (int)MyJoinOperation.Subtraction, in0.Count, in1.Count);
                        //ZXC m_dotKernel.Run(out0, 0, Owner.Temp, Owner.Temp, Owner.Temp.Count, /* distributed: */ 0);
                        m_dotKernel.Run(out0, Owner.Temp, Owner.Temp);
                        break;

                    case MyJoinOperation.CosineDistance:
                    case MyJoinOperation.DotProduct:
                        //ZXC m_dotKernel.Run(out0, 0, in0, in1, in0.Count, /* distributed: */ 0);
                        m_dotKernel.Run(out0, in0, in1);
                        break;

                    case MyJoinOperation.MatMultiplication:
                        m_kernel.Run(in0, in1, out0, in0.ColumnHint, in1.ColumnHint, out0.Count);
                        break;

                    default:
                        if (Owner.InputBranches == 2)
                            m_kernel.Run(in0, in1, out0, (int)Owner.Operation, in0.Count, in1.Count);
                        else
                            m_kernel.Run(
                                Owner.InputBlocksPointers,
                                Owner.InputBranches,
                                out0,
                                (int)Owner.Operation,
                                out0.Count);
                        break;
                }
            }
        }
    }
}
