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

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    public class MyStackLayer : MyAbstractLayer, IMyCustomTaskFactory
    {
        [MyInputBlock(1)]
        public MyMemoryBlock<float> Input1
        {
            get { return GetInput(1); }
        }

        // pointer to both layers
        new internal MyAbstractLayer[] PreviousLayer { get; set; }

        public override ConnectionType Connection
        {
            get { return ConnectionType.FULLY_CONNECTED; }
        }
        
        public MyStackLayer()
        {
            
        }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            int totalOutputs = 0;
            Output.ColumnHint = 1;

            PreviousLayer = new MyAbstractLayer[InputBranches];
            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> input = GetInput(i);

                if (input == null)
                    continue;

                PreviousLayer[i] = input.Owner as MyAbstractLayer;
                totalOutputs += input.Count;

                if (Output.ColumnHint == 1 && input.ColumnHint > 1)
                    Output.ColumnHint = input.ColumnHint;
            }

            // StackInputs operation
            Output.Count = totalOutputs;
        }

        public override string Description
        {
            get
            {
                return "Stack layer";
            }
        }
        
        public void CreateTasks()
        {
            ForwardTask = new MyStackForwardTask();
            DeltaBackTask = new MyStackBackDeltaTask();
        }

        [Description("DeltaBackTask"), MyTaskInfo(OneShot = false)]
        public class MyStackBackDeltaTask : MyTask<MyStackLayer>
        {
            public MyStackBackDeltaTask() { } //parameterless constructor

            public override void Init(int nGPU) { }

            public override void Execute() //Task execution
            {
                // propagate delta
                Owner.Delta.CopyFromMemoryBlock(Owner.NextLayer.Delta, 0, 0, Owner.NextLayer.Delta.Count);

                for (int i = 0; i < Owner.InputBranches; i++)
                    Owner.PreviousLayer[i].Delta.CopyFromMemoryBlock(Owner.Delta, 0, 0, Owner.Delta.Count);
            }
        }

        [Description("ForwardTask"), MyTaskInfo(OneShot = false)]
        public class MyStackForwardTask : MyTask<MyStackLayer>
        {
            public MyStackForwardTask() { } //parameterless constructor

            public override void Init(int nGPU) { }

            public override void Execute() //Task execution
            {
                int totalOutputs = 0;
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    MyMemoryBlock<float> input = Owner.GetInput(i);

                    if (input == null)
                        continue;

                    input.CopyToMemoryBlock(Owner.Output, 0, totalOutputs, input.Count);
                    totalOutputs += input.Count;
                }
                Owner.Output.SafeCopyToHost();
            }
        }
    }
}
