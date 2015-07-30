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
    public class MyStackLayer : MyAbstractWeightLayer, IMyCustomTaskFactory
    {
        [MyInputBlock(1)]
        public MyMemoryBlock<float> Input1
        {
            get { return GetInput(1); }
        }

        public override ConnectionType Connection
        {
            get { return ConnectionType.FULLY_CONNECTED; }
        }
        
        //public MyStackInputsTask StackInputs { get; private set; }

        public MyStackLayer()
        {
            //InputBranches = 2;
            this.ActivationFunction = ActivationFunctionType.NO_ACTIVATION;
        }


        public override void UpdateMemoryBlocks()
        {
            int totalOutputs = 0;
            Output.ColumnHint = 1;

            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);

                if (ai == null)
                    continue;

                totalOutputs += ai.Count;

                if (Output.ColumnHint == 1 && ai.ColumnHint > 1)
                {
                    Output.ColumnHint = ai.ColumnHint;
                }
            }

            base.UpdateMemoryBlocks();

            if (Neurons > 0)
            {
                if (Input != null)
                {
                    // parameter allocations
                    Weights.Count = Neurons * Input.Count;
                    Bias.Count = Neurons;

                    // SGD allocations
                    Delta.Count = Neurons;
                    PreviousWeightDelta.Count = Neurons * Input.Count; // momentum method
                    PreviousBiasDelta.Count = Neurons; // momentum method

                    // RMSProp allocations
                    MeanSquareWeight.Count = Weights.Count;
                    MeanSquareBias.Count = Bias.Count;
                }
            }

            // StackInputs operation
            Output.Count = totalOutputs;
            Neurons = totalOutputs;
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
            //StackInputs = new MyStackInputsTask();
        }

        [Description("DeltaBackTask"), MyTaskInfo(OneShot = false)]
        public class MyStackBackDeltaTask : MyTask<MyAbstractLayer>
        {
            public MyStackBackDeltaTask() { } //parameterless constructor

            public override void Init(int nGPU)
            {

            }

            public override void Execute() //Task execution
            {
                // pointer to both layers
                MyAbstractLayer previousLayer = Owner.PreviousLayer;
                MyAbstractLayer nextLayer = Owner.NextLayer;

                if (!(Owner.PreviousLayer is MyAbstractLayer) || !(Owner.PreviousLayer is MyAbstractLayer))
                    return;

                if (previousLayer != null && nextLayer != null)
                {
                    // propagate delta
                    //nextLayer.Delta.CopyToMemoryBlock(previousLayer.Delta, 0, 0, nextLayer.Delta.Count);
                    Owner.Delta.CopyFromMemoryBlock(nextLayer.Delta, 0, 0, nextLayer.Delta.Count);
                    //Owner.Delta.CopyFromMemoryBlock(previousLayer.Delta, 0, 0, previousLayer.Delta.Count);

                    previousLayer.Delta.SafeCopyToHost();
                    Owner.Delta.SafeCopyToHost();
                    nextLayer.Delta.SafeCopyToHost();
                }
            }
        }

        [Description("ForwardTask"), MyTaskInfo(OneShot = false)]
        public class MyStackForwardTask : MyTask<MyAbstractLayer>
        {
            public MyStackForwardTask() { } //parameterless constructor

            private MyCudaKernel m_forwardKernel; // kernel
            MyMemoryBlock<float> in0, in1, out0;

            public override void Init(int nGPU)
            {
                in0 = Owner.GetInput(0);
                in1 = Owner.GetInput(1);
                out0 = Owner.GetOutput(0);
                m_forwardKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernels", "OneToOneForwardKernel");
            }

            public override void Execute() //Task execution
            {
                int totalOutputs = 0;
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    MyMemoryBlock<float> ai = Owner.GetInput(i);

                    if (ai == null)
                        continue;

                    ai.CopyToMemoryBlock(Owner.Output, 0, totalOutputs, ai.Count);
                    totalOutputs += ai.Count;
                }
                Owner.Output.SafeCopyToHost();
            }
        }
    }
}
