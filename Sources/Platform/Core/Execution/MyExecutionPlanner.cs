using GoodAI.Core.Nodes;
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
        MyExecutionPlan CreateExecutionPlan(MyProject project, IEnumerable<MyWorkingNode> initNodes = null);
        MyExecutionBlock CreateNodeExecutionPlan(MyWorkingNode node, bool initPhase);
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

        /// <summary>
        /// Creates the execution plan.
        /// </summary>
        /// <param name="project">The whole project from which the standard execution plan will be built.</param>
        /// <param name="initNodes">Ordered list of new nodes from which the initialization plan will be built.</param>
        /// <returns>The created execution plan.</returns>
        public MyExecutionPlan CreateExecutionPlan(MyProject project, IEnumerable<MyWorkingNode> initNodes = null)
        {
            MyExecutionPlan executionPlan = new MyExecutionPlan();            

            IMyOrderingAlgorithm ordering = new MyHierarchicalOrdering();
            ordering.EvaluateOrder(project.Network);            

            var initBlocks = new List<IMyExecutable>();
            if (initNodes != null)
                initBlocks.AddRange(initNodes.Select(node => CreateNodeExecutionPlan(node, true)));

            executionPlan.InitStepPlan = new MyExecutionBlock(initBlocks.ToArray())
            { Name = "Initialization" };

            executionPlan.StandardStepPlan = new MyExecutionBlock(
                CreateNodeExecutionPlan(project.World, false),
                CreateNodeExecutionPlan(project.Network, false))
            { Name = "Simulation" };

            return executionPlan;
        }

        public MyExecutionBlock CreateNodeExecutionPlan(MyWorkingNode node, bool initPhase)
        {
            List<IMyExecutable> defaultPlanContent = new List<IMyExecutable>();

            if (!initPhase && PlanSignalTasks)
            {
                defaultPlanContent.Add(new MyIncomingSignalTask(node));
            }

            foreach (var taskPropertyInfo in node.GetInfo().TaskOrder)
            {
                MyTask task = node.GetTaskByPropertyName(taskPropertyInfo.Name);

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

            var defaultPlan = new MyExecutionBlock(defaultPlanContent.ToArray())
            {
                Name = node.Name
            };

            MyExecutionBlock resultPlan = defaultPlan;

            IMyCustomExecutionPlanner executionPlannerNode = node as IMyCustomExecutionPlanner;
            if (executionPlannerNode != null)
            {
                resultPlan = initPhase
                    ? executionPlannerNode.CreateCustomInitPhasePlan(defaultPlan)
                    : executionPlannerNode.CreateCustomExecutionPlan(defaultPlan);

                resultPlan.Name = defaultPlan.Name;
            }

            if (resultPlan.Name == null)
                resultPlan.Name = node.GetType().Name;

            if (node is MyNodeGroup)
            {
                resultPlan.Name += " (group)";
            }

            // TODO(HonzaS): Rethink this. It's only used in profiling results.
            node.ExecutionBlock = resultPlan;

            return resultPlan;
        }
    }
}
