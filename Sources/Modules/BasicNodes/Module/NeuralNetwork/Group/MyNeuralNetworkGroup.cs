using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.LSTM.Tasks;
using GoodAI.Modules.NeuralNetwork.Layers;
using GoodAI.Modules.NeuralNetwork.Tasks;
using GoodAI.Modules.RBM;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Group
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Network node group.</summary>
    /// <description>
    /// The Neural Network Group is necessary to build a neural network consisting of neural layers.<br></br>
    /// It is required to control the data flow during feed-forward and backpropagation between layers, as well as holding important hyperparameters and method variables.
    /// </description>
    public class MyNeuralNetworkGroup : MyNodeGroup, IMyCustomExecutionPlanner
    {
        // global rand which should be used by layers
        public static Random rng = new Random();

        // Signals
        // IsLearning is added to network input and set to true
        // it can be used in combination with output layers as
        // learning barriers which are useful for pretraining
        //public MyIsLearningSignal IsLearning { get; set; }
        //public class MyIsLearningSignal : MySignal { }

        //Node properties
        [ReadOnly(true)]
        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("\tTemporal")]
        public int TimeStep { get; set; }

        [YAXSerializableField(DefaultValue = 1)]
        [MyBrowsable, Category("\tTemporal")]
        public int SequenceLength { get; set; }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tRegularization")]
        public float L1 { get; set; }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tRegularization")]
        public float L2 { get; set; }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tRegularization")]
        public float Dropout { get; set; }

        //Memory Blocks
        public List<MyNode> SortedChildren;
        public MyAbstractLayer FirstTopologicalLayer;
        internal MyAbstractLayer LastTopologicalLayer;
        internal int TotalWeights;

        //Tasks
        [MyTaskGroup("BackPropagation")]
        public MySGDTask SGD { get; protected set; }
        [MyTaskGroup("BackPropagation")]
        public MyRMSTask RMS { get; protected set; }
        [MyTaskGroup("BackPropagation")]
        public MyAdadeltaTask Adadelta { get; protected set; }

        public MyAbstractBackpropTask GetActiveBackpropTask() {
            if (SGD.Enabled)
                return SGD;
            if (RMS.Enabled)
                return RMS;
            if (Adadelta.Enabled)
                return Adadelta;
            return null;
        }

        //[MyTaskGroup("BackPropagation")]
        //public MyvSGDfdTask vSGD { get; protected set; }

        public MyInitNNGroupTask InitGroup { get; protected set; }
        public MyIncrementTimeStepTask IncrementTimeStep { get; protected set; }
        public MyDecrementTimeStepTask DecrementTimeStep { get; protected set; }
        public MyRunTemporalBlocksModeTask RunTemporalBlocksMode { get; protected set; }
        public MyGradientCheckTask GradientCheck { get; protected set; }

        public MyNeuralNetworkGroup()
        {
            InputBranches = 2; // usually 2 inputs (input, target or input, reward)
            OutputBranches = 1; // usually 1 output (output or action)
        }  //parameterless constructor

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();
        }

        public virtual MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            return defaultInitPhasePlan;
        }

        public virtual MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            List<IMyExecutable> selected = new List<IMyExecutable>();
            List<IMyExecutable> newPlan = new List<IMyExecutable>();

            List<IMyExecutable> BPTTSingleStep = new List<IMyExecutable>();
            List<IMyExecutable> BPTTAllSteps = new List<IMyExecutable>();
            
            // copy default plan content to new plan content
            foreach (IMyExecutable groupTask in defaultPlan.Children)
                if (groupTask is MyExecutionBlock)
                    foreach (IMyExecutable nodeTask in (groupTask as MyExecutionBlock).Children)
                        newPlan.Add(nodeTask); // add individual node tasks
                else
                    newPlan.Add(groupTask); // add group tasks

            // remove group backprop tasks (they should be called from the individual layers)
            // DO NOT remove RBM tasks
            // DO NOT remove the currently selected backprop task (it handles batch learning)
            selected = newPlan.Where(task => task is MyAbstractBackpropTask &&  !(task.Enabled) && !(task is MyRBMLearningTask || task is MyRBMReconstructionTask)).ToList();
            newPlan.RemoveAll(selected.Contains);
            // bbpt single step
            BPTTSingleStep.AddRange(newPlan.Where(task => task is IMyDeltaTask).ToList().Reverse<IMyExecutable>());
            BPTTSingleStep.AddRange(newPlan.Where(task => task is MyLSTMPartialDerivativesTask).ToList());
            BPTTSingleStep.AddRange(newPlan.Where(task => task is MyQLearningTask).ToList());
            BPTTSingleStep.AddRange(newPlan.Where(task => task is MyGradientCheckTask).ToList());
            BPTTSingleStep.AddRange(newPlan.Where(task => task is MyRestoreValuesTask).ToList());
            BPTTSingleStep.AddRange(newPlan.Where(task => task is MySaveActionTask).ToList());
            BPTTSingleStep.Add(DecrementTimeStep);

            // backprop until unfolded (timestep=0)
            MyExecutionBlock BPTTLoop = new MyLoopBlock(i => TimeStep != -1,
                BPTTSingleStep.ToArray()
            );

            // bptt architecture
            BPTTAllSteps.Add(BPTTLoop);
            BPTTAllSteps.Add(RunTemporalBlocksMode);
            BPTTAllSteps.AddRange(newPlan.Where(task => task is IMyUpdateWeightsTask).ToList());

            // if current time is time for bbp, do it
            MyExecutionBlock BPTTExecuteBPTTIfTimeCountReachedSequenceLength = new MyIfBlock(() => TimeStep == SequenceLength-1,
                BPTTAllSteps.ToArray()
            );

            // remove group backprop tasks (they should be called from the individual layers)
            newPlan.RemoveAll(newPlan.Where(task => task is MyAbstractBackpropTask && !(task is MyRBMLearningTask || task is MyRBMReconstructionTask)).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyCreateDropoutMaskTask).ToList().Contains);
            //newPlan.RemoveAll(newPlan.Where(task => task is IMyOutputDeltaTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is IMyDeltaTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyGradientCheckTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is IMyUpdateWeightsTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyQLearningTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyRestoreValuesTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyLSTMPartialDerivativesTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MySaveActionTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyIncrementTimeStepTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyDecrementTimeStepTask).ToList().Contains);
            newPlan.RemoveAll(newPlan.Where(task => task is MyRunTemporalBlocksModeTask).ToList().Contains);
            
            selected = newPlan.Where(task => task is IMyOutputDeltaTask).ToList();
            newPlan.RemoveAll(selected.Contains);
    
            // after FF add deltaoutput and bptt if needed, then increpement one step :)
            newPlan.Insert(0, IncrementTimeStep);
            newPlan.InsertRange(newPlan.IndexOf(newPlan.FindLast(task => task is IMyForwardTask)) + 1, selected.Reverse<IMyExecutable>());
            newPlan.Add(BPTTExecuteBPTTIfTimeCountReachedSequenceLength);

            // return new plan as MyExecutionBlock
            return new MyExecutionBlock(newPlan.ToArray());
        }

        public void FeedForward()
        {
            MyAbstractLayer layer = FirstTopologicalLayer;
            while (layer != null)
            {
                layer.ForwardTask.Execute();
                layer = layer.NextTopologicalLayer;
            }
        }

        public float GetError()
        {
            // get the error from output layer
            if (LastTopologicalLayer is MyAbstractOutputLayer)
            {
                // pointer to output layer
                MyAbstractOutputLayer outputLayer = LastTopologicalLayer as MyAbstractOutputLayer;

                // get enabled loss function
                MyTask lossTask = outputLayer.GetEnabledTask("LossFunctions");

                // no loss function?
                if (lossTask == null)
                {
                    // Get call stack
                    StackTrace stackTrace = new StackTrace();

                    MyLog.ERROR.WriteLine("ERROR: GetError() called from " + stackTrace.GetFrame(1).GetMethod().Name + " needs a LossFunction task to be selected in the OutputLayer.");
                    return 0.0f;
                }

                // execute loss function
                lossTask.Execute();

                // copy to host
                outputLayer.Cost.SafeCopyToHost();

                // return cost (error)
                return outputLayer.Cost.Host[0];
            }
            else
            {
                // Get call stack
                StackTrace stackTrace = new StackTrace();

                MyLog.ERROR.WriteLine("ERROR: GetError() called from " + stackTrace.GetFrame(1).GetMethod().Name + " needs an OutputLayer as the last layer.");
                return 0.0f;
            }
        }

        // handles batch learning
        // should be called after every backward pass
        public void NextSample()
        {
            if (GetActiveBackpropTask() != null)
                GetActiveBackpropTask().BatchIndex++;
        }

        // are we at the beginning of a new batch - should we reset deltas?
        public bool NewBatch()
        {
            return (GetActiveBackpropTask() != null) && (GetActiveBackpropTask().BatchIndex == 0);
        }
    }
}