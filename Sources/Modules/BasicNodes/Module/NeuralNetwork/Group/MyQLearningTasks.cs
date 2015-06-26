using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using BrainSimulator.Task;
using System.Collections;
using System.ComponentModel;

using YAXLib;
using ManagedCuda;
using BrainSimulator.NeuralNetwork.Layers;

namespace BrainSimulator.NeuralNetwork.Group
{
    [Description("InitQLGroup"), MyTaskInfo(OneShot = true)]
    public class MyInitQLGroupTask : MyTask<MyQLearningGroup>
    {
        public MyInitQLGroupTask() { } //parameterless constructor
        public override void Init(int nGPU) { } //Kernel initialization

        public override void Execute() //Task execution
        {
            // set names of input & output branches
            if (Owner.InputBranches == 2 && Owner.OutputBranches == 1)
            {
                // can this be moved to the constructor? it's not working here!
                Owner.GroupInputNodes[0].Name = "State";
                Owner.GroupInputNodes[1].Name = "Reward";
                Owner.GroupOutputNodes[0].Name = "Action";
            }

            // set FirstLayer
            Owner.FirstLayer = null;
            foreach (MyNode child in Owner.SortedChildren)
            {
                if (child is MyLayer)
                {
                    Owner.FirstLayer = child as MyLayer;
                    break;
                }
            }

            // set LastLayer
            MyLayer selectedLayer = Owner.FirstLayer;
            Owner.LastLayer = selectedLayer;
            while (selectedLayer.NextLayer != null)
            {
                selectedLayer = selectedLayer.NextLayer as MyLayer; // this could potentionally break, if an abstract layer is present (which it should not be)!!!
                Owner.LastLayer = selectedLayer;
            }

            // set last layer offsets
            Owner.LastLayerOffset1D = Owner.AccSize1D - Owner.LastLayer.Neurons;
            Owner.LastLayerOffset2D = Owner.AccSize2D - Owner.LastLayer.Neurons * Owner.LastLayer.Input.Count;
        }
    }

    [Description("SaveToReplayMem"), MyTaskInfo(OneShot = false)]
    public class MySaveToReplayMemTask : MyTask<MyQLearningGroup>
    {
        public MySaveToReplayMemTask() { } //parameterless constructor
        public override void Init(int nGPU) { } //Kernel initialization

        public override void Execute() //Task execution
        {
            // set recording index
            Owner.RecIndex = (int)SimulationStep % Owner.TimestepsSaved;

            // record state, reward & action
            Owner.ReplayState.CopyFromMemoryBlock(Owner.State, 0, Owner.RecIndex * Owner.State.Count, Owner.State.Count);
            Owner.ReplayReward.CopyFromMemoryBlock(Owner.Reward, 0, Owner.RecIndex * Owner.Reward.Count, Owner.Reward.Count);
            Owner.ReplayAction.CopyFromMemoryBlock(Owner.Action, 0, Owner.RecIndex * Owner.Action.Count, Owner.Action.Count);
        }
    }

    [Description("CloneNetwork"), MyTaskInfo(OneShot = false)]
    public class MyCloneNetworkTask : MyTask<MyQLearningGroup>
    {
        //Properties
        [YAXSerializableField(DefaultValue = 1000)]
        [MyBrowsable, Category("\tParams")]
        public int CloneEach { get; set; }

        public MyCloneNetworkTask() { } //parameterless constructor
        public override void Init(int nGPU) { } //Kernel initialization

        public override void Execute() //Task execution
        {
            // clone network at the specified interval
            if (SimulationStep % CloneEach == 0)
            {
                int offset1D = 0;
                int offset2D = 0;
                MyLayer selectedLayer = Owner.FirstLayer;
                while (selectedLayer != null)
                {
                    int size1D = selectedLayer.Neurons;
                    int size2D = selectedLayer.Input.Count * selectedLayer.Neurons;
                    Owner.CloneWeights.CopyFromMemoryBlock(selectedLayer.Weights, 0, offset2D, size2D);
                    Owner.CloneBias.CopyFromMemoryBlock(selectedLayer.Bias, 0, offset1D, size1D);
                    offset1D += size1D;
                    offset2D += size2D;
                    selectedLayer = selectedLayer.NextLayer as MyLayer; // this could potentially break, if there is an abstract layer present (which there should not be)
                }
                if (Owner.SGD.Momentum != 0)
                {
                    Owner.ClonePreviousBiasDelta.Fill(0);
                    Owner.ClonePreviousWeightDelta.Fill(0);
                }
            }
        }
    }

    [Description("LearnFromReplayMem"), MyTaskInfo(OneShot = false)]
    public class MyLearnFromReplayMemTask : MyTask<MyQLearningGroup>
    {
        //Properties
        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tParams")]
        public int ReplaysEachTimestep { get; set; }

        [YAXSerializableField(DefaultValue = 0.99f)]
        [MyBrowsable, Category("\tParams")]
        public float DiscountFactor { get; set; }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tParams")]
        public bool BindTargetValues { get; set; }

        [YAXSerializableField(DefaultValue = 1.0f)]
        [MyBrowsable, Category("\tParams")]
        public float UpperBound { get; set; }

        [YAXSerializableField(DefaultValue = -1.0f)]
        [MyBrowsable, Category("\tParams")]
        public float LowerBound { get; set; }

        public MyLearnFromReplayMemTask() { } //parameterless constructor

        //additional kernels
        private MyCudaKernel m_kernel;
        private MyCudaKernel m_outputDeltaKernel;
        private MyCudaKernel m_hiddenDeltaKernel;
        private MyCudaKernel m_updateWeightsKernel;
        private MyCudaKernel m_updateWeightsL2Kernel;


        //Kernel initialization
        public override void Init(int nGPU)
        {
            m_kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\FeedForwardKernel", "FeedForwardKernel");
            m_outputDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\OutputDeltaKernel", "outputDeltaKernel");
            m_hiddenDeltaKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\HiddenDeltaKernel", "hiddenDeltaKernel");
            m_updateWeightsKernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\UpdateWeightsKernel", "FullyConnectedSGDUpdateKernel");
            m_updateWeightsL2Kernel = MyKernelFactory.Instance.Kernel(nGPU, @"NeuralNetwork\Layer\UpdateWeightsKernel", "updateWeightsWithL2Regularization");
        }

        public override void Execute() //Task execution
        {
            if (SimulationStep > Owner.TimestepsSaved) // start replaying when ReplayMemory is full
            {
                // reset delta - batch learning
                Owner.CloneDelta.Fill(0);

                for (int r = 0; r < ReplaysEachTimestep; r++)
                {
                    // set t_0
                    int t_0 = Owner.rnd.Next(0, Owner.TimestepsSaved);
                    if (t_0 == Owner.RecIndex) t_0++; // last recorded memory not allowed (no following timestep recorded)
                    if (t_0 == Owner.TimestepsSaved) t_0 = 0; // don't exceed the memory

                    // set t_1
                    int t_1 = t_0 + 1;
                    if (t_1 == Owner.TimestepsSaved) t_1 = 0; // don't exceed the memory

                    // get action(t_0) taken
                    int actionTaken = GetAction(t_0);

                    // init target to reward at t_0 - (actually t_1 since it works better for our pong) - maybe this should be possible to set dynamically?
                    Owner.ReplayReward.SafeCopyToHost(t_1, 1);
                    float target = Owner.ReplayReward.Host[t_1];

                    // add t_1 discounted value to target unless this is a terminal state (negative reward)
                    if (target >= 0)
                    {
                        // add the highest t_1 Qvalue discounted to target
                        CloneFeedForward(t_1);
                        Owner.CloneOutput.SafeCopyToHost(Owner.LastLayerOffset1D, Owner.LastLayer.Neurons); // copy to host

                        target += DiscountFactor * Owner.CloneOutput.Host.Skip(Owner.LastLayerOffset1D).Max(); // target is the best value from next state
                        //target += DiscountFactor * Owner.CloneOutput.Host[Owner.LastLayerOffset1D + actionTaken]; // has to repeat action
                    }

                    // init target values to t_0 value for all actions
                    CloneFeedForward(t_0);
                    Owner.CloneTarget.CopyFromMemoryBlock(Owner.CloneOutput, Owner.LastLayerOffset1D, 0, Owner.LastLayer.Neurons); // copy to CloneTarget

                    // set target value for actionTaken
                    Owner.CloneTarget.Host[actionTaken] = target;
                    Owner.CloneTarget.SafeCopyToDevice(actionTaken, 1);

                    // bind all target values
                    if (BindTargetValues)
                    {
                        Owner.CloneTarget.SafeCopyToHost();
                        for (int a = 0; a < Owner.Action.Count; a++)
                        {
                            if (Owner.CloneTarget.Host[a] > UpperBound)
                                Owner.CloneTarget.Host[a] = UpperBound;
                            else if (Owner.CloneTarget.Host[a] < LowerBound)
                                Owner.CloneTarget.Host[a] = LowerBound;
                        }
                        Owner.CloneTarget.SafeCopyToDevice();
                    }

                    // calculate deltas
                    CloneCalcDeltas();
                }

                // update weights - batch learning
                UpdateWeights();
            }
        }

        private void FeedForward(int replayIndex)
        {
            Owner.CloneState.CopyFromMemoryBlock(Owner.ReplayState, replayIndex * Owner.CloneState.Count, 0, Owner.CloneState.Count);

            MyLayer selectedLayer = Owner.FirstLayer;
            Owner.CloneInput = Owner.CloneState.GetDevicePtr(Owner); // set CloneInput ptr to CloneState
            int offset1D = 0;
            int offset2D = 0;
            while (selectedLayer != null)
            {
                //this will setup thread dimensions (or you can set it on the kernel itself)
                m_kernel.SetupExecution(selectedLayer.Neurons);

                //runs a kernel with given parameters
                m_kernel.Run(
                    (int)selectedLayer.ActivationFunction,
                    Owner.CloneInput,
                    Owner.CloneOutput.GetDevicePtr(Owner.GPU, offset1D),
                    selectedLayer.Weights,
                    Owner.CloneWeightedInput.GetDevicePtr(Owner.GPU, offset1D),
                    selectedLayer.Bias,
                    selectedLayer.Input.Count,
                    selectedLayer.Output.Count
                    );

                offset1D += selectedLayer.Neurons;
                offset2D += selectedLayer.Input.Count * selectedLayer.Neurons;

                Owner.CloneInput = selectedLayer.Output.GetDevicePtr(Owner.GPU);
                selectedLayer = selectedLayer.NextLayer as MyLayer; // this could potentially break, if there is an abstract layer present (which there should not be)
            }
        }

        private void CloneFeedForward(int replayIndex)
        {
            Owner.CloneState.CopyFromMemoryBlock(Owner.ReplayState, replayIndex * Owner.CloneState.Count, 0, Owner.CloneState.Count);

            MyLayer selectedLayer = Owner.FirstLayer;
            Owner.CloneInput = Owner.CloneState.GetDevicePtr(Owner); // set CloneInput ptr to CloneState
            int offset1D = 0;
            int offset2D = 0;
            while (selectedLayer != null)
            {
                //this will setup thread dimensions (or you can set it on the kernel itself)
                m_kernel.SetupExecution(selectedLayer.Neurons);

                //runs a kernel with given parameters
                m_kernel.Run(
                    (int)selectedLayer.ActivationFunction,
                    Owner.CloneInput,
                    Owner.CloneOutput.GetDevicePtr(Owner.GPU, offset1D),
                    Owner.CloneWeights.GetDevicePtr(Owner.GPU, offset2D),
                    Owner.CloneWeightedInput.GetDevicePtr(Owner.GPU, offset1D),
                    Owner.CloneBias.GetDevicePtr(Owner.GPU, offset1D),
                    selectedLayer.Input.Count,
                    selectedLayer.Output.Count
                    );

                Owner.CloneInput = Owner.CloneOutput.GetDevicePtr(Owner.GPU, offset1D);

                offset1D += selectedLayer.Neurons;
                offset2D += selectedLayer.Input.Count * selectedLayer.Neurons;

                selectedLayer = selectedLayer.NextLayer as MyLayer; // this could potentially break, if there is an abstract layer present (which there should not be)
            }
        }

        private void CloneCalcDeltas()
        {
            int offset1D = Owner.LastLayerOffset1D;
            int offset2D = Owner.LastLayerOffset2D;

            // calculate output layer delta
            m_outputDeltaKernel.SetupExecution(Owner.LastLayer.Neurons);
            m_outputDeltaKernel.Run(
                (int)Owner.LastLayer.ActivationFunction,
                Owner.CloneOutput.GetDevicePtr(Owner.GPU, offset1D),
                Owner.CloneTarget,
                Owner.CloneDelta.GetDevicePtr(Owner.GPU, offset1D),
                Owner.CloneWeightedInput.GetDevicePtr(Owner.GPU, offset1D),
                Owner.LastLayer.Neurons
                );

            // calculate deltas for hidden layers
            MyLayer selectedLayer = Owner.LastLayer.PreviousLayer as MyLayer; // this could potentially break, if there is an abstract layer present (which there should not be)
            while (selectedLayer != null)
            {
                // calculate hidden layer delta
                m_hiddenDeltaKernel.SetupExecution(selectedLayer.Neurons);
                m_hiddenDeltaKernel.Run(
                    (int)selectedLayer.ActivationFunction,
                    Owner.CloneWeightedInput.GetDevicePtr(Owner.GPU, offset1D - selectedLayer.Neurons),
                    Owner.CloneDelta.GetDevicePtr(Owner.GPU, offset1D - selectedLayer.Neurons),
                    Owner.CloneDelta.GetDevicePtr(Owner.GPU, offset1D),
                    Owner.CloneWeights.GetDevicePtr(Owner.GPU, offset2D),
                    selectedLayer.Neurons,
                    selectedLayer.NextLayer.Neurons
                    );

                offset1D -= selectedLayer.Neurons;
                offset2D -= selectedLayer.Neurons * selectedLayer.Input.Count;

                selectedLayer = selectedLayer.PreviousLayer as MyLayer; // this could potentially break, if there is an abstract layer present (which there should not be)
            }
        }

        private void UpdateWeights()
        {
            int offset1D = 0;
            int offset2D = 0;

            // update weights
            MyLayer selectedLayer = Owner.FirstLayer;
            Owner.CloneInput = Owner.CloneState.GetDevicePtr(Owner); // set CloneInput ptr to CloneState
            while (selectedLayer != null)
            {
                m_updateWeightsKernel.SetupExecution(selectedLayer.Neurons);
                m_updateWeightsKernel.Run(
                    Owner.CloneInput,
                    Owner.CloneDelta.GetDevicePtr(Owner.GPU, offset1D),
                    selectedLayer.Weights, // update the original weights, not the cloned ones
                    selectedLayer.PreviousWeightDelta,
                    selectedLayer.Bias,
                    selectedLayer.PreviousBiasDelta,
                    Owner.SGD.TrainingRate,
                    Owner.SGD.Momentum,
                    selectedLayer.Input.Count,
                    selectedLayer.Neurons
                    );

                offset1D += selectedLayer.Neurons;
                offset2D += selectedLayer.Input.Count * selectedLayer.Neurons;

                Owner.CloneInput = selectedLayer.Output.GetDevicePtr(Owner.GPU);
                selectedLayer = selectedLayer.NextLayer as MyLayer; // this could potentially break, if there is an abstract layer present (which there should not be)
            }
        }

        private int GetAction(int replayIndex)
        {
            int actionTaken = 0;
            int actionIndex = replayIndex * Owner.Action.Count;
            float actionValue = float.NegativeInfinity;
            Owner.ReplayAction.SafeCopyToHost(actionIndex, Owner.Action.Count);
            for (int a = 0; a < Owner.Action.Count; a++)
            {
                if (Owner.ReplayAction.Host[actionIndex + a] > actionValue)
                {
                    actionValue = Owner.ReplayAction.Host[actionIndex + a];
                    actionTaken = a;
                }
            }
            return actionTaken;
        }
    }
}