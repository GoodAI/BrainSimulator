using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using ManagedCuda;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    /// <author>GoodAI</author>
    /// <tag>#mm</tag>
    /// <status>Working</status>
    /// <summary>
    ///   Static symbol table shared between all nodes of the same SymbolSize.
    ///   Use Mode to specify whether to generate a symbol or test it through dot product with the input.
    /// </summary>
    /// <description>
    ///   Pre-computes vectors that are shared between nodes of this type that have the same SymbolSize.
    /// </description>
    public class MyCodeBook : MyCodeBookBase
    {
        public enum CodeBookMode
        {
            GenerateOutput,
            CompareWithInput
        }


        #region Memory blocks

        public MyMemoryBlock<float> TempBlock { get; private set; }

        #endregion

        #region Properties

        [MyBrowsable, Category("Function")]
        [YAXSerializableField]
        [Description("Specifies whether the output should be a constant random vector or if it should be compared to the input.")]
        public CodeBookMode Mode { get; set; }

        [MyBrowsable, Category("Function")]
        [YAXSerializableField(DefaultValue = false)]
        [Description("Specifies if the input should be regarded as an offset added to the currently selected CodeVector.")]
        public bool InputAsOffset { get; set; }

        #endregion

        #region MyNode overrides

        public override void Validate(MyValidator validator)
        {
            if (Mode == CodeBookMode.CompareWithInput)
            {
                base.Validate(validator);
                if (Input != null)
                {
                    validator.AssertError(SymbolSize == Input.Count, this, "Input size differs from output size.");
                }
            }
            else
            {
                if (InputAsOffset)
                {
                    base.Validate(validator);
                    if (Input != null)
                    {
                        validator.AssertError(Input.Count == 1, this, "Input size must be exactly 1.");
                    }
                }
                validator.AssertError(SymbolSize > 0, this, "Non zero output size is required.");
            }
        }

        public override void UpdateMemoryBlocks()
        {
            if (Mode == CodeBookMode.CompareWithInput)
            {
                if (UseBSCVariety)
                {
                    TempBlock.Count = SymbolSize;
                    TempBlock.ColumnHint = ColumnHint;
                }

                Output.Count = 1;
                Output.ColumnHint = 1;
            }
            else
            {
                Output.Count = SymbolSize;
                Output.ColumnHint = ColumnHint;
            }
        }

        public override string Description
        {
            get
            {
                return (Mode == CodeBookMode.CompareWithInput ? "Is \"" + PerformTask.CodeVector + "\"?" : "\"" + PerformTask.CodeVector + "\"");
            }
        }

        #endregion


        public MyCodeVectorsTask PerformTask { get; private set; }


        /// <summary>
        ///   Select pre-computed random vectors by using the CodeVector property.
        /// </summary>
        [Description("Perform Task")]
        public class MyCodeVectorsTask : MyTask<MyCodeBook>
        {
            private static MyJoin.MyJoinOperation _similarityOperator = MyJoin.MyJoinOperation.XNOR;

            private MyReductionKernel<float> m_sum;
            private MyProductKernel<float> m_dot;
            private MyCudaKernel m_similarityKernel;

            private int lastIdx = -1;


            [MyBrowsable, Category("Params")]
            [YAXSerializableField(DefaultValue = MyCodeVector.Empty)]
            public MyCodeVector CodeVector { get; set; }

            [MyBrowsable, Category("SHARED: Binary Spatter Code")]
            [YAXSerializableField(DefaultValue = MyJoin.MyJoinOperation.XNOR)]
            public MyJoin.MyJoinOperation SimilarityOperator
            {
                get { return _similarityOperator; }
                set { _similarityOperator = value; }
            }


            public override void Init(int nGPU)
            {
                lastIdx = -1;

                if (Owner.UseBSCVariety)
                    m_sum = MyKernelFactory.Instance.KernelReduction<float>(Owner, nGPU, ReductionMode.f_Sum_f);
                else
                    m_dot = MyKernelFactory.Instance.KernelProduct<float>(Owner, nGPU, ProductMode.f_DotProduct_f);

                MyMemoryManager.Instance.ClearGlobalVariable(Owner.GlobalVariableName, nGPU);

                if (Owner.UseBSCVariety)
                {
                    m_similarityKernel = MyKernelFactory.Instance.Kernel(Owner.GPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernel");
                    m_similarityKernel.SetupExecution(Owner.SymbolSize);
                }
            }

            public override void Execute()
            {
                int index = (int)CodeVector;

                if (Owner.Mode == CodeBookMode.GenerateOutput && Owner.InputAsOffset)
                {
                    Owner.Input.SafeCopyToHost();
                    int offset = (int)Owner.Input.Host[0];

                    if (index + offset < typeof(MyCodeVector).GetEnumValues().Length)
                    {
                        index += offset;
                    }
                }

                CudaDeviceVariable<float> codeVector = new CudaDeviceVariable<float>(
                    MyMemoryManager.Instance.GetGlobalVariable<float>(Owner.GlobalVariableName, Owner.GPU, Owner.GenerateRandomVectors).DevicePointer
                    + index * Owner.SymbolSize * sizeof(float), Owner.SymbolSize);

                if (Owner.Mode == CodeBookMode.GenerateOutput)
                {
                    if (lastIdx != index)
                        Owner.Output.GetDevice(Owner).CopyToDevice(codeVector, 0, 0, sizeof(float) * Owner.SymbolSize);
                }
                else
                {
                    if (Owner.UseBSCVariety)
                    {
                        m_similarityKernel.Run(codeVector.DevicePointer, Owner.Input, Owner.TempBlock, (int)SimilarityOperator, Owner.SymbolSize);
                        //ZXC m_sum.Run(Owner.Output, Owner.TempBlock, Owner.SymbolSize, 0, 0, 1, /* distributed: */ 0);
                        m_sum.size = Owner.SymbolSize;
                        m_sum.Run(Owner.Output, Owner.TempBlock);
                    }
                    else
                    {
                        //ZXC m_dot.Run(Owner.Output, 0, codeVector.DevicePointer, Owner.Input, Owner.SymbolSize, /* distributed: */ 0);
                        m_dot.Run(Owner.Output, codeVector.DevicePointer, Owner.Input.GetDevicePtr(Owner), Owner.SymbolSize);
                    }
                }

                lastIdx = index;
            }
        }
    }
}
