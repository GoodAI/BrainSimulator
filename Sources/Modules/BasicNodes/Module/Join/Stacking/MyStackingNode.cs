using GoodAI.Core.Memory;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.Transforms;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core;
using GoodAI.Core.Nodes;
using YAXLib;

namespace GoodAI.Modules.Join
{
    /// <author>GoodAI</author>
    /// <tag>#mm+mp</tag>
    /// <status>Working</status>
    /// <summary>
    /// Joins two or more vectors into a single vector.
    /// </summary>
    /// <description>
    /// <ul>
    /// <li><b>Concatenate:</b> Places each successive vector after the end of the previous vector.</li>
    /// <li><b>Interweave:</b> Interprets input vectors as matrices (based on their ColumnHint) and concatenates the 
    ///     rows of the successive matrices. It concatenates the first rows of the matrices, then it concatenates the 
    ///     second rows of the matrices etc. Inputs must have the same number of rows.</li>
    /// </ul>
    /// </description>
    public class MyStackingNode : MyWorkingNode, IMyVariableBranchViewNodeBase
    {
        #region Memory blocks

        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        #endregion

        #region Properties

        [ReadOnly(false)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set { base.InputBranches = value; }
        }

        [MyBrowsable, YAXSerializableField(DefaultValue = 0), YAXElementFor("IO")]
        public int OutputColHint { get; set; }

        [MyBrowsable, Category("Behavior")]
        [YAXSerializableField(DefaultValue = MyStackingOperation.Concatenate), YAXElementFor("Behavior")]
        public MyStackingOperation Operation { get; set; }

        public int OutputSize
        {
            get { return Output.Count; }
            set { Output.Count = value; }
        }

        #endregion


        private int[] _offsets = new int[0];
        private MyMemoryBlock<float>[] _inputBlocks;


        public MyStackingNode()
        {
            InputBranches = 2;
        }


        #region MyWorkingNode overrides

        public override void UpdateMemoryBlocks()
        {
            if (_offsets.Length < InputBranches)
            {
                _offsets = new int[InputBranches];
                _inputBlocks = new MyMemoryBlock<float>[InputBranches];
            }

            int outputSize = 0;
            Output.ColumnHint = 1;

            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);
                _inputBlocks[i] = ai;

                if (ai == null)
                    continue;

                _offsets[i] = outputSize;
                outputSize += ai.Count;

                if (Output.ColumnHint == 1 && ai.ColumnHint > 1)
                {
                    Output.ColumnHint = ai.ColumnHint;
                }
            }

            if (OutputColHint > 0)
            {
                Output.ColumnHint = OutputColHint;
            }

            OutputSize = outputSize;


            switch (Operation)
            {
                case MyStackingOperation.Concatenate:
                case MyStackingOperation.Interweave:
                    break;
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (validator.ValidationSucessfull)
            {
                string errorOutput;
                validator.AssertError(MyStackingOps.Validate(Operation,_inputBlocks, Output, out errorOutput), this, errorOutput);
            }
        }

        public override string Description { get { return Operation.ToString(); } }

        #endregion


        public MyStackInputsTask StackInputs { get; private set; }

        /// <summary>
        ///   Performs the desired join operation.
        /// </summary>
        [Description("Perform join operation")]
        public class MyStackInputsTask : MyTask<MyStackingNode>
        {
            private MyStackingOps _stackingOps;


            public override void Init(int nGPU)
            {
                _stackingOps = new MyStackingOps(Owner, Owner.Operation);
            }

            public override void Execute()
            {
                _stackingOps.Run(Owner.Operation, Owner.Output, Owner._inputBlocks);
            }
        }
    }
}
