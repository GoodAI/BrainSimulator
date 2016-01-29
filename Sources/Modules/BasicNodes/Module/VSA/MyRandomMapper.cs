using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Matrix;
using GoodAI.Modules.Transforms;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    /// <author>GoodAI</author>
    /// <tag>#mm</tag>
    /// <status>Working</status>
    /// <summary>
    ///   Performs a random projection to another dimension by computing A.x with a random matrix A and the input vector x.
    ///   The transformation matrix is shared with nodes that do the same transformation (including its inverse, see <see cref="DoDecoding"/>)
    ///   and have the same <see cref="NameGroup"/>. You can select which matrix axis to normalize to produce vectors of different lengths.
    /// </summary>
    /// <description></description>
    public class MyRandomMapper : MyRandomPool
    {
        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> Input { get { return GetInput(0); } }


        [MyUnmanaged]
        public MyMemoryBlock<float> UnmanagedVectors { get; set; }

        [MyUnmanaged]
        public MyMemoryBlock<float> UnmanagedBaseVectors { get; set; }

        public MyMemoryBlock<float> Temp { get; set; }

        #endregion

        #region Properties

        [MyBrowsable, Category("Matrix generation")]
        [Description("Nodes doing the same projection with the same NameGroup share seed and transformation matrix.")]
        [YAXSerializableField(DefaultValue = 0)]
        public int NameGroup { get; set; }

        [MyBrowsable, Category("Structure")]
        [Description("Specifies, if the inverse of the transformation matrix should be used.")]
        [YAXSerializableField(DefaultValue = false)]
        public bool DoDecoding { get; set; }

        [MyBrowsable, Category("Structure")]
        [Description("Specifies the output dimension of the transformation matrix (its number of rows).")]
        [YAXSerializableField(DefaultValue = 1024)]
        public int OutputSize { get; set; }


        int InputSize { get { return Input != null ? Input.Count : 0; } }
        bool ChangeDim { get { return PerformTask != null && PerformTask.Enabled; } }
        bool ChangeDimDoQuery { get { return ChangeDim && DoDecoding; } }

        protected string GlobalVariableBasesName { get { return GlobalVariableName + "bases"; } }

        #endregion

        #region MyRandomPool overrides

        protected override string GlobalVariableName
        {
            get
            {
                return "RANDOM_SYMBOL_POOL_" + NameGroup + '_'
                    + '_' + (ChangeDimDoQuery
                                ? (OutputSize + "->" + InputSize)
                                : (InputSize + "->" + OutputSize));
            }
        }

        protected override int PatternCount { get { return 0; } }

        public override int Seed { get { return 12345 + NameGroup; } }

        #endregion

        #region MyNode overrides

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(OutputSize > 0, this, "Output must be greater than 0.");

            if (Input != null)
            {
                if (!DoDecoding)
                    validator.AssertError(OutputSize == SymbolSize, this, "Symbol size must be equal to output size");
                else
                    validator.AssertError(InputSize == SymbolSize, this, "Symbol size must be equal to input size.");
            }
        }

        public override string Description { get { return "f(x) = R" + (ChangeDimDoQuery ? "\u0027" : string.Empty) + " \u00B7 x"; } }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = OutputSize;
            Output.ColumnHint = ColumnHint;

            int longerDim = Math.Max(InputSize, OutputSize);

            UnmanagedBaseVectors.Count = VectorMode == VectorGenerationMode.AverageBaseVectors ? longerDim * longerDim : 0;
            UnmanagedBaseVectors.ColumnHint = longerDim;

            UnmanagedVectors.Count = InputSize * OutputSize;
            UnmanagedVectors.ColumnHint = InputSize;

            Temp.Count = longerDim;

            MyMemoryManager.Instance.ClearGlobalVariable(GlobalVariableName, GPU);
            MyMemoryManager.Instance.ClearGlobalVariable(GlobalVariableBasesName, GPU);
        }

        #endregion


        public MyGenerateMatrixTask InitTask { get; private set; }
        public MyRandomMapperTask PerformTask { get; private set; }


        /// <summary>
        ///   Allocates all the memory needed to store and create the desired transformation matrices and generates them based on the selected settings.
        /// </summary>
        [Description("Generate projection vectors"), MyTaskInfo(OneShot = true, Disabled = false)]
        public class MyGenerateMatrixTask : MyTask<MyRandomMapper>
        {
            private bool _first, _firstBase;


            [MyBrowsable, Category("Structure")]
            [YAXSerializableField(DefaultValue = AxisToNormalizeEnum.xDim)]
            public AxisToNormalizeEnum AxisToNormalize { get; set; }


            public override void Init(int nGPU)
            {
                _first = _firstBase = false;

                if (Owner.VectorMode == VectorGenerationMode.AverageBaseVectors)
                    Owner.UnmanagedBaseVectors.ExternalPointer = MyMemoryManager.Instance.GetGlobalVariable(Owner.GlobalVariableBasesName, nGPU,
                        () =>
                        {
                            _firstBase = true;
                            int longerDim = Math.Max(Owner.InputSize, Owner.OutputSize);
                            return new float[longerDim * longerDim];
                        }).DevicePointer.Pointer;

                Owner.UnmanagedVectors.ExternalPointer = MyMemoryManager.Instance.GetGlobalVariable(Owner.GlobalVariableName, Owner.GPU,
                    () =>
                    {
                        _first = true;
                        return new float[Owner.InputSize * Owner.OutputSize];
                    }).DevicePointer.Pointer;
            }

            public override void Execute()
            {
                // Let only one node init the shared thing
                if ((Owner.VectorMode == VectorGenerationMode.AverageBaseVectors && !_firstBase) || !_first)
                    return;


                var random = new Random(Owner.Seed);

                var dotKernel = MyKernelFactory.Instance.KernelProduct<float>(Owner, Owner.GPU, ProductMode.f_DotProduct_f);
                var multKernel = MyKernelFactory.Instance.Kernel(Owner.GPU, @"common\CombineVectorsKernel", "CombineTwoVectorsKernelVarSize");
                var transposeKernel = MyKernelFactory.Instance.Kernel(Owner.GPU, @"VSA\Mappers", "Transpose");

                int xDim, yDim;
                AxisToNormalizeEnum axisToNormalize = AxisToNormalize;

                if (!Owner.DoDecoding)
                {
                    xDim = Owner.InputSize;
                    yDim = Owner.OutputSize;
                }
                else
                {
                    xDim = Owner.OutputSize;
                    yDim = Owner.InputSize;
                    axisToNormalize = axisToNormalize == AxisToNormalizeEnum.xDim
                        ? AxisToNormalizeEnum.yDim
                        : AxisToNormalizeEnum.xDim;
                }

                GenerateTransformMatrix(
                    Owner.UnmanagedVectors, Owner.UnmanagedBaseVectors, Owner.Temp,
                    random, xDim, yDim,
                    dotKernel, multKernel, transposeKernel, Owner.GPU,
                    Owner.VectorMode, axisToNormalize);

                MyMemoryManager.Instance.ClearGlobalVariable(Owner.GlobalVariableBasesName, Owner.GPU);
                MyMemoryManager.Instance.RemoveBlock(Owner, Owner.UnmanagedBaseVectors);
            }
        }


        /// <summary>
        ///   Transforms the input by the transformation matrix as specified by the parameters.
        /// </summary>
        [Description("Map to another dimension")]
        public class MyRandomMapperTask : MyTask<MyRandomMapper>
        {
            private MyMatrixCublasOps ops;


            public override void Init(int nGPU)
            {
                ops = new MyMatrixCublasOps(Owner, MatOperation.Multiplication);
            }

            public override void Execute()
            {
                var x = Owner.Input.GetDevice(Owner);
                var A = MyMemoryManager.Instance.GetGlobalVariable(Owner.GlobalVariableName, Owner.GPU, () => new float[0]); // The initializer should never be called here
                var y = Owner.Output.GetDevice(Owner);


                int xDim = Owner.InputSize;
                int yDim = Owner.OutputSize;

                // Transform the input to have OutputSize dimensions
                // If this is query, the matrix was created by the non-query node, it is thus transposed
                // Transposition is needed because of legacy reasons
                if (!Owner.DoDecoding)
                    // Non-query -- transposed multiplication
                    ops.Run(MatOperation.Multiplication, x, xDim, xDim, A, xDim * yDim, yDim, y, yDim, yDim, 0);
                else
                    // Query mode -- non-transposed multiplication
                    ops.Run(MatOperation.Multiplication, A, xDim * yDim, xDim, x, xDim, 1, y, yDim, 1, 0);
            }
        }
    }
}
