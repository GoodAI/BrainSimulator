using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.VSA
{
    /// <author>GoodAI</author> 
    /// <tag>#mm</tag>
    /// <status>Working</status>
    /// <summary>
    ///   Provides the Bind and Unbind operations for HRR or BSC symbols, as well permuting. The implementation for HRR is based on the fast Fourier transformation.
    /// </summary>
    public class MyBindingNode : MyWorkingNode
    {
        public enum BindingMode
        {
            HRR,
            BSC,
            Permute,
        }


        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> FirstInput
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> SecondInput
        {
            get { return GetInput(1); }
        }

        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public MyMemoryBlock<float> Temp { get; private set; }

        #endregion

        #region Properties

        [MyBrowsable, Category("General")]
        [YAXSerializableField(DefaultValue = BindingMode.HRR)]
        public BindingMode Mode { get; set; }

        public int InputSize
        {
            get { return FirstInput != null ? FirstInput.Count : 0; }
        }

        #endregion

        #region MyNode overrides

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            if (FirstInput != null && SecondInput != null)
            {
                validator.AssertError(SecondInput.Count % FirstInput.Count == 0, this, "Operand sizes differ!");
            }
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = InputSize;
            Output.ColumnHint = FirstInput != null ? FirstInput.ColumnHint : 1;

            Temp.Count = MyFourierBinder.GetTempBlockSize(InputSize);
            Temp.ColumnHint = Output.ColumnHint;
        }

        public override string Description
        {
            get
            {
                switch (Mode)
                {
                    case BindingMode.HRR:
                        return "f(x,y) = " + (BindInputs.DoQuery ? "x ⊕ y" : "x ⊗ y");

                    case BindingMode.BSC:
                        return "f(x,y) = x ⊕ y";

                    case BindingMode.Permute:
                        return "f(x,\u03c0) = " + (BindInputs.DoQuery ? "\u03c0\u207b\u00b9(x)" : "\u03c0(x)");

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        #endregion


        public MyBindingTask BindInputs { get; private set; }


        /// <summary>
        ///   Performs the binding or unbinding operation.
        /// </summary>
        [Description("BindInternal inputs")]
        public class MyBindingTask : MyTask<MyBindingNode>
        {
            private MySymbolBinderBase m_binder;


            [MyBrowsable, Category("Unbind")]
            [YAXSerializableField]
            [Description("Do binding with the inverse of the second parameter.")]
            public bool DoQuery { get; set; }

            [MyBrowsable, Category("Unbind")]
            [YAXSerializableField]
            [Description("Use an exact inverse instead of involution when unbinding.")]
            public bool UseExactQuery { get; set; }

            [MyBrowsable, Category("Normalization")]
            [YAXSerializableField]
            [Description("Normalize the resulting vector.")]
            public bool ExactNormalization { get; set; }

            [MyBrowsable, Category("Normalization")]
            [YAXSerializableField(DefaultValue = false)]
            [Description("Specifies, if the manual Denominator should be used. Only effective when ExactNormalization is not true.")]
            public bool ManualDenomination { get; set; }

            [MyBrowsable, Category("Normalization")]
            [YAXSerializableField(DefaultValue = 1)]
            [Description("The resulting vector will be multiplied by this value. Only effective when ManualDenomination is true and ExactNormalization is false.")]
            public float Denominator { get; set; }


            public override void Init(int nGPU)
            {
                switch (Owner.Mode)
                {
                    case BindingMode.HRR:
                        m_binder = new MyFourierBinder(Owner, Owner.InputSize, Owner.Temp);
                        break;
                    case BindingMode.BSC:
                        m_binder = new MyXORBinder(Owner, Owner.InputSize);
                        break;
                    case BindingMode.Permute:
                        m_binder = new MyPermutationBinder(Owner, Owner.InputSize, Owner.Temp);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            public override void Execute()
            {
                m_binder.NormalizeOutput = ExactNormalization;
                m_binder.ExactQuery = UseExactQuery;

                if (ManualDenomination && Denominator != 0)
                {
                    m_binder.Denominator = Denominator;
                }

                if (!DoQuery)
                {
                    m_binder.Bind(Owner.FirstInput, Owner.SecondInput, Owner.Output);
                }
                else
                {
                    m_binder.Unbind(Owner.FirstInput, Owner.SecondInput, Owner.Output);
                }
            }
        }
    }
}
