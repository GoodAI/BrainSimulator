using BrainSimulator.Execution;
using BrainSimulator.Memory;
using BrainSimulator.NeuralNetwork.Group;
using BrainSimulator.NeuralNetwork.Layers;
using BrainSimulator.NeuralNetwork.Tasks;
using BrainSimulator.Nodes;
using BrainSimulator.Task;
using BrainSimulator.Utils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace BrainSimulator.NeuralNetwork.Group
{
    /// <author>Philip Hilm</author>
    /// <status>WIP</status>
    /// <summary>QLearning group.</summary>
    /// <description>
    ///     Group with the functionality and execution planning needed for QLearning.
    ///     Derived from NeuralNetworkGroup
    /// </description>
    public class MyQLearningGroup : MyNeuralNetworkGroup
    {
        [ReadOnly(true)]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set { base.InputBranches = value; }
        }

        [ReadOnly(true)]
        public override int OutputBranches
        {
            get { return base.OutputBranches; }
            set { base.OutputBranches = value; }
        }

        public MyQLearningGroup() //parameterless constructor
        {
            // set inputs and outputs
            InputBranches = 2;
            OutputBranches = 1;
        }

        //Node properties
        [YAXSerializableField(DefaultValue = 1000)]
        [MyBrowsable, Category("\tReplayMemory")]
        public int TimestepsSaved { get; set; }

        // Memory blocks
        public MyMemoryBlock<float> ReplayState { get; private set; }
        public MyMemoryBlock<float> ReplayReward { get; private set; }
        public MyMemoryBlock<float> ReplayAction { get; private set; }

        public MyMemoryBlock<float> CloneWeightedInput { get; protected set; }
        public MyMemoryBlock<float> CloneDelta { get; protected set; }
        public MyMemoryBlock<float> CloneTarget { get; protected set; }

        [MyPersistable]
        public MyMemoryBlock<float> CloneWeights { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> ClonePreviousWeightDelta { get; protected set; }

        [MyPersistable]
        public MyMemoryBlock<float> CloneBias { get; protected set; }
        [MyPersistable]
        public MyMemoryBlock<float> ClonePreviousBiasDelta { get; protected set; }

        public MyMemoryBlock<float> CloneState { get; protected set; }
        public CUdeviceptr CloneInput { get; set; }
        public MyMemoryBlock<float> CloneOutput { get; protected set; }

        // test
        public MyMemoryBlock<float> DynamicTrainingRate { get; protected set; }

        internal int AccSize1D; // accumulated neurons
        internal int AccSize2D; // accumulated input x neurons

        internal int RecIndex;
        internal MyMemoryBlock<float> State;
        internal MyMemoryBlock<float> Reward;
        internal MyMemoryBlock<float> Action;

        internal MyLayer FirstLayer;
        internal MyLayer LastLayer;
        internal int LastLayerOffset1D;
        internal int LastLayerOffset2D;

        internal Random rnd = new Random();

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            // calculate and set accumulated memory
            AccSize1D = 0;
            AccSize2D = 0;
            foreach (MyNode node in Children)
            {
                if (node is MyLayer)
                {
                    MyLayer layer = node as MyLayer;
                    if (layer.Input != null)
                    {
                        AccSize1D += layer.Neurons;
                        AccSize2D += layer.Input.Count * layer.Neurons;
                    }
                }
            }
            if (AccSize1D > 0)
            {
                CloneWeightedInput.Count = AccSize1D;
                CloneDelta.Count = AccSize1D;
                CloneBias.Count = AccSize1D;
                CloneOutput.Count = AccSize1D;
                ClonePreviousBiasDelta.Count = AccSize1D;

                // allocate memory scaling with input
                CloneWeights.Count = AccSize2D;
                ClonePreviousWeightDelta.Count = AccSize2D;
            }

            // make sure we have the correct number of inputs and outputs
            if (InputBranches == 2 && OutputBranches == 1)
            {
                int stateSize = GetInputSize(0);
                int rewardSize = GetInputSize(1);
                int actionSize = GetOutputSize(0);

                if (stateSize > 0)
                {
                    ReplayState.Count = stateSize * TimestepsSaved;
                    State = GetInput(0);

                    CloneState.Count = stateSize;
                }

                if (rewardSize > 0)
                {
                    ReplayReward.Count = rewardSize * TimestepsSaved;
                    Reward = GetInput(1);
                }

                if (actionSize > 0)
                {
                    ReplayAction.Count = actionSize * TimestepsSaved;
                    Action = GetOutput(0);

                    CloneTarget.Count = actionSize;
                }
            }
        }

        //Tasks
        public MyInitQLGroupTask InitQLGroup { get; private set; }
        public MySaveToReplayMemTask SaveToReplayMem { get; private set; }
        public MyLearnFromReplayMemTask LearnFromReplayMem { get; private set; }
        public MyCloneNetworkTask CloneNetwork { get; private set; }

        public override MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            defaultInitPhasePlan = base.CreateCustomExecutionPlan(defaultInitPhasePlan);

            List<IMyExecutable> NewPlan = new List<IMyExecutable>();

            // execute MyInitQLGroupTask last
            MyInitQLGroupTask initTask = null;
            foreach (IMyExecutable groupTask in defaultInitPhasePlan.Children)
                if (groupTask is MyInitQLGroupTask)
                    initTask = groupTask as MyInitQLGroupTask;
                else
                    NewPlan.Add(groupTask);
            NewPlan.Add(initTask);

            return new MyExecutionBlock(NewPlan.ToArray());
        }

        public override MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            defaultPlan = base.CreateCustomExecutionPlan(defaultPlan);

            List<IMyExecutable> selected = new List<IMyExecutable>();
            List<IMyExecutable> newPlan = new List<IMyExecutable>();

            // copy default plan content to new plan content
            foreach (IMyExecutable groupTask in defaultPlan.Children)
                if (groupTask is MyExecutionBlock)
                    foreach (IMyExecutable nodeTask in (groupTask as MyExecutionBlock).Children)
                        newPlan.Add(nodeTask); // add individual node tasks
                else
                    newPlan.Add(groupTask); // add group tasks

            // select & remove CalcDeltaTask(s) & UpdateWeightsTask(s)
            selected = newPlan.Where(task => task is MyCalcDeltaTask).ToList();
            selected.AddRange(newPlan.Where(task => task is MyUpdateWeightsTaskDeprecated).ToList());
            newPlan.RemoveAll(selected.Contains);

            // move MySaveToReplayMemTask to the end of the list
            selected = newPlan.Where(task => task is MySaveToReplayMemTask).ToList();
            newPlan.RemoveAll(selected.Contains);
            newPlan.AddRange(selected);

            // move MyLearnFromReplayMemTask to the end of the list
            selected = newPlan.Where(task => task is MyLearnFromReplayMemTask).ToList();
            newPlan.RemoveAll(selected.Contains);
            newPlan.AddRange(selected);

            // return new plan as MyExecutionBlock
            return new MyExecutionBlock(newPlan.ToArray());
        }

        //Validation rules
        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            //validator.AssertError(Input != null, this, "No input available");
        }
    }
}