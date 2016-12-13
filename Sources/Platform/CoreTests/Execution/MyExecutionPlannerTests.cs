using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using Xunit;

namespace CoreTests.Execution
{
    public class MyExecutionPlannerTests
    {
        [Fact]
        public void PlannerRespectsTaskOrder()
        {
            var planner = new MyDefaultExecutionPlanner();
            
            var taskOrderNode = new TaskOrderNode();

            MyNodeInfo.CollectNodeInfo(taskOrderNode.GetType());

            MyExecutionBlock execBlock = planner.CreateNodeExecutionPlan(taskOrderNode, initPhase: false);
           
            var lastOrder = -1;
            foreach (IMyExecutable task in execBlock.Children)
            {
                var taskInfo = task.GetType().GetCustomAttribute<MyTaskInfoAttribute>(true);
                if (taskInfo == null)
                    continue;

                Assert.True(lastOrder <= taskInfo.Order,
                    "Task order must be greater or equal to the previous task order");

                lastOrder = taskInfo.Order;
            }
        }

        [Fact]
        public void PlannerKeepsOrderOfTasksWithoutOrderAttribute()
        {
            var planner = new MyDefaultExecutionPlanner();
            
            var unorderedTasksNode = new UnorderedTasksNode();

            MyNodeInfo.CollectNodeInfo(unorderedTasksNode.GetType());

            MyExecutionBlock execBlock = planner.CreateNodeExecutionPlan(unorderedTasksNode, initPhase: false);
           
            // This is maybe a bit fragile. Remove the test if it breaks due to unrelated and intended changes.
            Assert.Equal("CherryTask", execBlock.Children[1].Name);
            Assert.Equal("BananaTask", execBlock.Children[2].Name);
            Assert.Equal("AppleTask", execBlock.Children[3].Name);
        }
    }
}
