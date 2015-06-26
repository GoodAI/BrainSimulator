using BrainSimulator;
using BrainSimulator.Nodes;
using BrainSimulator.Memory;
using BrainSimulator.Utils;
using BrainSimulator.Task;
using BrainSimulator.Execution;
using BrainSimulator.Signals;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Threading.Tasks;
using YAXLib;
using ManagedCuda;
using BrainSimulator.NeuralNetwork.Layers;
using BrainSimulator.NeuralNetwork.Tasks;
using System.Diagnostics;

namespace BrainSimulator.NeuralNetwork.Group
{
    /// <author>Philip Hilm</author>
    /// <status>WIP</status>
    /// <summary>Network node group.</summary>
    /// <description></description>
    public class MyNeuralNetworkGroup : MyNodeGroup, IMyCustomExecutionPlanner
    {
        //Node properties
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
        internal MyAbstractLayer FirstLayer;
        internal MyAbstractLayer LastLayer;
        internal int TotalWeights;

        //Tasks
        [MyTaskGroup("BackPropagation")]
        public MySGDTask SGD { get; protected set; }
        [MyTaskGroup("BackPropagation")]
        public MyRMSTask RMS { get; protected set; }
        [MyTaskGroup("BackPropagation")]
        public MyvSGDfdTask vSGD { get; protected set; }

        public MyInitNNGroupTask InitGroup { get; protected set; }
        public MyGradientCheckTask GradientCheck { get; protected set; }

        public MyNeuralNetworkGroup() { }  //parameterless constructor

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

            // copy default plan content to new plan content
            foreach (IMyExecutable groupTask in defaultPlan.Children)
                if (groupTask is MyExecutionBlock)
                    foreach (IMyExecutable nodeTask in (groupTask as MyExecutionBlock).Children)
                        newPlan.Add(nodeTask); // add individual node tasks
                else
                    newPlan.Add(groupTask); // add group tasks

            // remove group backprop tasks (they should be called from the individual layers)
            selected = newPlan.Where(task => task is MyAbstractBackpropTask).ToList();
            newPlan.RemoveAll(selected.Contains);

            // move MyCreateDropoutMaskTask(s) before the first MyForwardTask
            selected = newPlan.Where(task => task is MyCreateDropoutMaskTask).ToList();
            newPlan.RemoveAll(selected.Contains);
            newPlan.InsertRange(newPlan.IndexOf(newPlan.Find(task => task is IMyForwardTask)), selected);

            // move reversed MyOutputDeltaTask(s) after the last MyForwardTask (usually there is only one)
            selected = newPlan.Where(task => task is IMyOutputDeltaTask).ToList();
            newPlan.RemoveAll(selected.Contains);
            selected.Reverse();
            newPlan.InsertRange(newPlan.IndexOf(newPlan.FindLast(task => task is IMyForwardTask)) + 1, selected);

            // move reversed MyDeltaTask(s) after the last MyOutputDeltaTask
            selected = newPlan.Where(task => task is IMyDeltaTask).ToList();
            newPlan.RemoveAll(selected.Contains);
            selected.Reverse();
            newPlan.InsertRange(newPlan.IndexOf(newPlan.FindLast(task => task is IMyOutputDeltaTask)) + 1, selected);

            // move MyGradientCheckTask after the last MyDeltaTask
            selected = newPlan.Where(task => task is MyGradientCheckTask).ToList();
            newPlan.RemoveAll(selected.Contains);
            newPlan.InsertRange(newPlan.IndexOf(newPlan.FindLast(task => task is IMyDeltaTask)) + 1, selected);

            // move MyUpdateWeightsTask(s) after the last MyGradientCheckTask
            selected = newPlan.Where(task => task is IMyUpdateWeightsTask).ToList();
            newPlan.RemoveAll(selected.Contains);
            newPlan.InsertRange(newPlan.IndexOf(newPlan.FindLast(task => task is MyGradientCheckTask)) + 1, selected);

            // return new plan as MyExecutionBlock
            return new MyExecutionBlock(newPlan.ToArray());
        }

        public void FeedForward()
        {
            MyAbstractLayer layer = FirstLayer;
            while (layer != null)
            {
                layer.ForwardTask.Execute();
                layer = layer.NextLayer;
            }
        }

        public float GetError()
        {
            // get the error from output layer
            if (LastLayer is MyOutputLayer)
            {
                // pointer to output layer
                MyOutputLayer outputLayer = LastLayer as MyOutputLayer;

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
    }
}