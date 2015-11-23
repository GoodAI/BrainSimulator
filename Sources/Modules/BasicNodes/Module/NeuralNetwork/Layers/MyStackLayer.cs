using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System;
using System.ComponentModel;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    /// <author>GoodAI</author>
    /// <status>Working</status>
    /// <summary>
    ///   Node that stacks 2 input nodes and forwards them as one output. It can send deltas back from output to inputs while gradient descent is performed.
    /// </summary>
    /// <description>
    ///   A Stack-Layer node should be used as a subnode of the NeuralNetworkGroup.
    ///   It can join 2 nodes together (stack them) and forward them as one node.
    ///   Moreover, while a learning phase of the gradient descent is active and deltas are
    ///   back-prapagated to StackLayer than they are properly distributed to its input nodes.
    ///   Specifically, this is important when we need 2 neural-layers to work in parallel.
    /// </description>
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
            get { return ConnectionType.ONE_TO_ONE; }
        }

        [YAXSerializableField(DefaultValue = ActivationFunctionType.NO_ACTIVATION)]
        [MyBrowsable, Category("\tLayer"), ReadOnly(true)]
        public override ActivationFunctionType ActivationFunction
        {
            get { return ActivationFunctionType.NO_ACTIVATION; }
            set { }
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
            Neurons = totalOutputs / ParentNetwork.BatchSize;
        }

        public override string Description
        {
            get
            {
                return "Stack layer";
            }
        }

        public override void Validate(MyValidator validator)
        {
            for (int i = 0; i < InputBranches; i++)
            {
                MyMemoryBlock<float> ai = GetInput(i);

                if (ai == null)
                    validator.AddError(this, string.Format("Missing input {0}.", i));
            }
        }

        public override bool SupportsBatchLearning { get { return true; } }

        public void CreateTasks()
        {
            ForwardTask = new MyStackForwardTask();
            DeltaBackTask = new MyStackBackDeltaTask();
        }

        /// <summary>
        /// Sends deltas back
        /// </summary>
        [Description("DeltaBackTask"), MyTaskInfo(OneShot = false)]
        public class MyStackBackDeltaTask : MyAbstractBackDeltaTask<MyStackLayer>
        {
            public MyStackBackDeltaTask() { } //parameterless constructor

            private MyCudaKernel m_deltaKernel; // kernel

            public override void Init(int nGPU)
            {
                m_deltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\DeltaKernels", "PreActivationFunctionDeltaKernel");
            }

            public override void Execute() //Task execution
            {
                Owner.Delta.SafeCopyToHost();

                int deltaIdx = 0;
                for (int b = 0; b < Owner.ParentNetwork.BatchSize; b++)
                {
                    for (int i = 0; i < Owner.InputBranches; i++)
                    {
                        MyAbstractLayer prevLayer = Owner.PreviousLayer[i];
                        int singleInputSize = Owner.GetInput(i).Count / Owner.ParentNetwork.BatchSize;

                        if (prevLayer != null)
                        {
                            //Owner.Delta.CopyToMemoryBlock(prevLayer.Delta, deltaIdx, b * singleInputSize, singleInputSize); //copy this on cpu instead, since it is faster most of the time
                            Buffer.BlockCopy(Owner.Delta.Host, deltaIdx * sizeof(float), prevLayer.Delta.Host, b * singleInputSize * sizeof(float), singleInputSize * sizeof(float));
                        }

                        deltaIdx += singleInputSize;
                    }
                }

                //copy previous layers' delta memory blocks to device memory and backpropagate through activation function of previous layer
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    if (Owner.PreviousLayer[i] != null)
                    {
                        Owner.PreviousLayer[i].Delta.SafeCopyToDevice();

                        m_deltaKernel.SetupExecution(Owner.PreviousLayer[i].Neurons * Owner.ParentNetwork.BatchSize);
                        m_deltaKernel.Run(
                            (int)Owner.PreviousLayer[i].ActivationFunction,
                            MyAbstractLayer.DetermineInput(Owner.PreviousLayer[i]),
                            Owner.PreviousLayer[i].Delta,
                            Owner.PreviousLayer[i].Neurons,
                            Owner.ParentNetwork.BatchSize
                        );
                    }
                }
            }
        }

        /// <summary>
        /// Forwards inputs to output
        /// </summary>
        [Description("ForwardTask"), MyTaskInfo(OneShot = false)]
        public class MyStackForwardTask : MyAbstractForwardTask<MyStackLayer>
        {
            public MyStackForwardTask() { } //parameterless constructor

            public override void Init(int nGPU) { }

            public override void Execute() //Task execution
            {
                //copy input memory blocks to host memory
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    Owner.GetInput(i).SafeCopyToHost();
                }

                int outputIdx = 0;
                for (int b = 0; b < Owner.ParentNetwork.BatchSize; b++)
                {
                    for (int i = 0; i < Owner.InputBranches; i++)
                    {
                        int singleInputSize = Owner.GetInput(i).Count / Owner.ParentNetwork.BatchSize;

                        //input.CopyToMemoryBlock(Owner.Output, b * singleInputSize, outputIdx, singleInputSize); //copy this on cpu instead, since it is faster most of the time
                        Buffer.BlockCopy(Owner.GetInput(i).Host, b * singleInputSize * sizeof(float), Owner.Output.Host, outputIdx * sizeof(float), singleInputSize * sizeof(float));

                        outputIdx += singleInputSize;
                    }
                }

                Owner.Output.SafeCopyToDevice();
            }
        }
    }
}
