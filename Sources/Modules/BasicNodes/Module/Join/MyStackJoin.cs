using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.Join
{
    /// <author>Good AI</author>
    /// <tag>#mm</tag>
    /// <status>In progress</status>
    /// <summary>
    ///   Performs an element-wise stacking join operation on the input vectors with error backpropagation to previous layer.
    /// </summary>
    public class MyStackJoin : MyWorkingNode, IMyCustomTaskFactory
    {
        [MyOutputBlock]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }

        public int OutputSize
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

        [MyBrowsable, YAXSerializableField(DefaultValue = 0), YAXElementFor("IO")]
        public int OutputColHint { get; set; }

        public int[] m_offsets = new int[0];

        public MyMemoryBlock<CUdeviceptr> InputBlocksPointers { get; private set; }
        public MyMemoryBlock<float> Temp { get; private set; }

        public MyInitTask InitMemoryMapping { get; private set; }
        public MyStackInputsTask StackInputs { get; private set; }
        public MyFCBackDeltaTask DeltaBackTask { get; private set; }

        public MyStackJoin()
        {
            //Output.Count = 1;
            //InputBranches = 2;
            //UpdateMemoryBlocks();
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

            if (OutputColHint > 0)
            {
                Output.ColumnHint = OutputColHint;
            }

            // StackInputs operation
            OutputSize = totalOutputs;
            InputBlocksPointers.Count = InputBranches;
        }

        public override string Description
        {
            get
            {
                return "Stack join";
            }
        }
        
        public void CreateTasks()
        {
            DeltaBackTask = new MyFCBackDeltaTask();
        }

        /// <summary>
        ///   Initializes any memory needed to perform the join operation.
        /// </summary>
        [Description("Init memory mapping"), MyTaskInfo(OneShot = true)]
        public class MyInitTask : MyTask<MyStackJoin>
        {
            public override void Init(int nGPU) { }

            public override void Execute()
            {
                
            }
        }

        /// <summary>
        ///   Performs the desired join operation.
        /// </summary>
        [Description("Perform join operation")]
        public class MyStackInputsTask : MyTask<MyStackJoin>
        {
            MyMemoryBlock<float> in0, in1, out0;

            private MyCudaKernel m_kernel;

            public override void Init(int nGPU)
            {
                in0 = Owner.GetInput(0);
                in1 = Owner.GetInput(1);
                out0 = Owner.GetOutput(0);

                m_kernel = Owner.InputBranches > 2
                    ? MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel")
                    : MyKernelFactory.Instance.Kernel(nGPU, @"Common\CombineVectorsKernel", "CombineTwoVectorsKernelVarSize");

                m_kernel.SetupExecution(out0.Count);
            }

            public override void Execute()
            {
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    MyMemoryBlock<float> ai = Owner.GetInput(i);
                    if (ai != null)
                    {
                        out0.CopyFromMemoryBlock(ai, 0, Owner.m_offsets[i], ai.Count);
                    }
                }
            }
        }

    }
}
