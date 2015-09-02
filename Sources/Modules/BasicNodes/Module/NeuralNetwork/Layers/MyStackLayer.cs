using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System.ComponentModel;

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
        }

        /// <summary>
        /// Sends deltas back
        /// </summary>
        [Description("DeltaBackTask"), MyTaskInfo(OneShot = false)]
        public class MyStackBackDeltaTask : MyAbstractBackDeltaTask<MyStackLayer>
        {
            public MyStackBackDeltaTask() { } //parameterless constructor

            public override void Init(int nGPU) { }

            public override void Execute() //Task execution
            {
                // propagate delta
                //Owner.Delta.CopyFromMemoryBlock(Owner.NextLayer.Delta, 0, 0, Owner.NextLayer.Delta.Count);

                int totalInputs = 0;
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    MyMemoryBlock<float> input = Owner.GetInput(i);
                    MyAbstractLayer prevLayer = Owner.PreviousLayer[i];
                    if (prevLayer != null)
                    {
                        prevLayer.Delta.CopyFromMemoryBlock(Owner.Delta, totalInputs, 0, input.Count);
                    }
                    totalInputs += input.Count;
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
                int totalOutputs = 0;
                for (int i = 0; i < Owner.InputBranches; i++)
                {
                    MyMemoryBlock<float> input = Owner.GetInput(i);

                    if (input == null)
                        continue;

                    input.CopyToMemoryBlock(Owner.Output, 0, totalOutputs, input.Count);
                    totalOutputs += input.Count;
                }
            }   
        }
    }
}
