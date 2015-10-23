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
    public class MyDistanceNode : MyTransform
    {
        #region Memory blocks

        [MyInputBlock(0)]
        public MyMemoryBlock<float> A
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> B
        {
            get { return GetInput(1); }
        }


        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public MyMemoryBlock<float> Temp { get; set; }

        #endregion


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
                    if (A != null)
                        Temp.Count = A.Count;
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
                validator.AssertError(MyDistanceOps.Validate(Operation, A.Count, B.Count, Temp.Count, Output.Count, out errorOutput), this, errorOutput);
            }
        }

        public override string Description { get { return Operation.ToString(); } }

        #endregion


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
                _distOperation.Run(Owner.Operation, Owner.A, Owner.B, Owner.Output);
            }
        }
    }
}