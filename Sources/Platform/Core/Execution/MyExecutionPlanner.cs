﻿using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System.Collections.Generic;
using System.Linq;

namespace GoodAI.Core.Execution
{
    /// Plan for executing the tasks on GPUs
    public class MyExecutionPlan
    {
        public MyExecutionBlock InitStepPlan { get; set; }  // Plan used for first simulation step
        public MyExecutionBlock StandardStepPlan { get; set; }  // Plan used for all simulation steps except the first one
    }

    public interface IMyExecutionPlanner
    {
        MyExecutionPlan CreateExecutionPlan(MyProject project);        
    }

    public interface IMyCustomExecutionPlanner
    {
        MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan);
        MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan);
    } 

    public class MyDefaultExecutionPlanner : IMyExecutionPlanner
    {
        public bool PlanSignalTasks { get; set; }

        public MyDefaultExecutionPlanner()
        {
            PlanSignalTasks = true;
        }

        public MyExecutionPlan CreateExecutionPlan(MyProject project)
        {
            MyExecutionPlan executionPlan = new MyExecutionPlan();            

            IMyOrderingAlgorithm ordering = new MyHierarchicalOrdering();
            ordering.EvaluateOrder(project.Network);            

            executionPlan.InitStepPlan = new MyExecutionBlock(
                CreateNodeExecutionPlan(project.World, true), 
                CreateNodeExecutionPlan(project.Network, true));
            executionPlan.InitStepPlan.Name = "Initialization";

            executionPlan.StandardStepPlan = new MyExecutionBlock(
                CreateNodeExecutionPlan(project.World, false),
                CreateNodeExecutionPlan(project.Network, false));
            executionPlan.StandardStepPlan.Name = "Simulation";

            return executionPlan;
        }

        private MyExecutionBlock CreateNodeExecutionPlan(MyWorkingNode node, bool initPhase)
        {
            List<IMyExecutable> defaultPlanContent = new List<IMyExecutable>();

            if (!initPhase && PlanSignalTasks)
            {
                defaultPlanContent.Add(new MyIncomingSignalTask(node));
            }

            foreach (string taskName in node.GetInfo().KnownTasks.Keys)
            {
                MyTask task = node.GetTaskByPropertyName(taskName);

                if (task != null && !task.DesignTime && (initPhase && task.OneShot || !initPhase && !task.OneShot))
                {
                    defaultPlanContent.Add(task);
                }
            }

            MyNodeGroup nodeGroup = node as MyNodeGroup;
            if (nodeGroup != null)
            {
                IEnumerable<MyNode> children = nodeGroup.Children.OrderBy(x => x.TopologicalOrder);

                foreach (MyNode childNode in children)
                {
                    MyWorkingNode childWorkingNode = childNode as MyWorkingNode;
                    if (childWorkingNode != null)
                    {
                        defaultPlanContent.Add(CreateNodeExecutionPlan(childWorkingNode, initPhase));
                    }
                }
            }

            if (!initPhase && PlanSignalTasks)
            {
                defaultPlanContent.Add(new MyOutgoingSignalTask(node));
            }

            MyExecutionBlock defaultPlan = new MyExecutionBlock(defaultPlanContent.ToArray());
            defaultPlan.Name = node.Name;

            MyExecutionBlock resultPlan = defaultPlan;

            IMyCustomExecutionPlanner executionPlannerNode = node as IMyCustomExecutionPlanner;
            if (executionPlannerNode != null)
            {
                if (initPhase)
                {
                    resultPlan = executionPlannerNode.CreateCustomInitPhasePlan(defaultPlan);
                }
                else
                {
                    resultPlan = executionPlannerNode.CreateCustomExecutionPlan(defaultPlan);
                }
                resultPlan.Name = defaultPlan.Name;
            }

            if (node is MyNodeGroup)
            {
                resultPlan.Name += " (group)";
            }

            return resultPlan;
        }
    }

    /* NOT USABLE ANYMORE
    public class MyTaskwiseExecution : IMyExecutionStrategy
    {
        public MyExecutionBlock PlanExecution(List<MyWorkingNode> orderedNodes)
        {
            Dictionary<Type, List<MyTask>> taskTable = new Dictionary<Type,List<MyTask>>();
            List<Type> taskOrder = new List<Type>();            

            foreach (MyWorkingNode node in orderedNodes.OrderBy(x => x.TopologicalOrder))
            {
                foreach (Type taskType in node.KnownTasks)
                {
                    if (!taskTable.ContainsKey(taskType))
                    {
                        taskTable.Add(taskType, new List<MyTask>());
                        taskOrder.Add(taskType);
                    }
                    taskTable[taskType].Add(node.GetTask(taskType));
                }
            }

            List<MyTask> result = new List<MyTask>();
            taskOrder.ForEach(t => result.AddRange(taskTable[t]));

            return new MyExecutionBlock(result.ToArray());
        }
    }

    
    public class MySequentialFirstExecution : IMyExecutionStrategy
    {
        public List<MyTask> PlanExecution(List<MyWorkingNode> nodes)
        {
            List<MyTask> result = new List<MyTask>();

            Dictionary<Type, List<MyTask>> taskTable = new Dictionary<Type, List<MyTask>>();
            List<Type> taskOrder = new List<Type>();
            
            foreach (MyWorkingNode node in nodes.OrderBy(x => x.TopologicalOrder))
            {
                if (node.Sequential)
                {
                    foreach (Type taskType in node.KnownTasks)
                    {
                        result.Add(node.GetTask(taskType));
                    }
                }
                else
                {
                    foreach (Type taskType in node.KnownTasks)
                    {
                        if (!taskTable.ContainsKey(taskType))
                        {
                            taskTable.Add(taskType, new List<MyTask>());
                            taskOrder.Add(taskType);
                        }
                        taskTable[taskType].Add(node.GetTask(taskType));
                    }
                }
            }
            
            taskOrder.ForEach(t => result.AddRange(taskTable[t]));

            return result;
        }
    }
     * */
}
