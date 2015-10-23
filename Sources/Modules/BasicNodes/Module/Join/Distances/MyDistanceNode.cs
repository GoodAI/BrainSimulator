using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using YAXLib;
using GoodAI.Modules.Transforms;

namespace GoodAI.Modules.Join
{

    /// <author>GoodAI</author>
    /// <tag>#mm+mp</tag>
    /// <status> Working </status>
    /// <summary>
    /// Takes two vectors and outputs a single number as a measure of their similarity.
    /// </summary>
    /// <description>
    /// To process more vectors at once, use MatrixNode (not all DistanceNode's operations are supported, though).
    /// </description>
    public class MyDistanceNode : MyTransform, IMyVariableBranchViewNodeBase
    {
        #region Memory blocks

        // Output is inherited from MyTransform

        public MyMemoryBlock<float> Temp { get; set; }

        #endregion

        [ReadOnly(true)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set { base.InputBranches = value; }
        }

        [MyBrowsable, Category("Behavior"), YAXSerializableField(DefaultValue = DistanceOperation.DotProd), YAXElementFor("Behavior")]
        public DistanceOperation Operation { get; set; }


        #region MyWorkingNode overrides

        public override void UpdateMemoryBlocks()
        {
            switch (Operation)
            {
                case DistanceOperation.EuclidDist:
                case DistanceOperation.EuclidDistSquared:
                case DistanceOperation.HammingDist:
                case DistanceOperation.HammingSim:
                    if (GetInput(0) != null)
                        Temp.Count = GetInput(0).Count;
                    break;

                default:
                    Temp.Count = 0;
                    break;
            }

            Output.Count = 1;
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (validator.ValidationSucessfull)
            {
                string errorOutput;
                validator.AssertError(MyDistanceOps.Validate(Operation, GetInput(0).Count, GetInput(1).Count, Temp.Count, Output.Count, out errorOutput), this, errorOutput);
            }
        }

        public override string Description { get { return Operation.ToString(); } }

        #endregion

        public MyDistanceNode()
        {
            InputBranches = 2;
        }


        public MyExecuteTask Execute { get; private set; }

        /// <summary>
        /// Computes the distance of the input vectors.
        /// </summary>
        [Description("Execute")]
        public class MyExecuteTask : MyTask<MyDistanceNode>
        {
            private MyDistanceOps _distOperation;


            public override void Init(int nGPU)
            {
                _distOperation = new MyDistanceOps(Owner, Owner.Operation, Owner.Temp); // it may need A for setting up kernel size!
            }

            public override void Execute()
            {
                _distOperation.Run(Owner.Operation, Owner.GetInput(0), Owner.GetInput(1), Owner.Output);
            }
        }
    }
}