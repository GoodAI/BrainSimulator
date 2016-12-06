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

            MyNodeInfo.CollectNodeInfo(typeof(TaskOrderNode));

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
    }
}
