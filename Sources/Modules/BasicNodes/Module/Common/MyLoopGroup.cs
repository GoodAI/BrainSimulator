using GoodAI.Core.Execution;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.Common
{
    public class MyLoopGroup : MyNodeGroup, IMyCustomExecutionPlanner
    {
        [YAXSerializableField(DefaultValue = 3)]
        [MyBrowsable, Category("Iterations")]
        public int Iterations { get; set; }

        public virtual MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            return defaultInitPhasePlan;
        }

        public virtual MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            List<IMyExecutable> newPlan = new List<IMyExecutable>();

            // copy default plan content to new plan content
            foreach (IMyExecutable groupTask in defaultPlan.Children)
                if (groupTask is MyExecutionBlock)
                    foreach (IMyExecutable nodeTask in (groupTask as MyExecutionBlock).Children)
                        newPlan.Add(nodeTask); // add individual node tasks
                else
                    newPlan.Add(groupTask); // add group tasks

            List<IMyExecutable> completePlan = new List<IMyExecutable>();
            for (int i = 0; i < Iterations; ++i)
            {
                completePlan.AddRange(newPlan);
            }
            return new MyExecutionBlock(completePlan.ToArray());
        }
    }
}
