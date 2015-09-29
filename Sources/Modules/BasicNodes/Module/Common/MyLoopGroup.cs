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
    /// <author>GoodAI</author>
    /// <meta>mv</meta>
    /// <staus>Working</staus>
    /// <summary>
    /// Group that is ran multiple iterations at one simulation step. 
    /// </summary>
    /// <description></description>
    public class MyLoopGroup : MyNodeGroup, IMyCustomExecutionPlanner
    {
        [YAXSerializableField(DefaultValue = 3)]
        [MyBrowsable, Category("Iterations")]
        public int Iterations { get; set; }

        public enum MyLoopOperation
        {
            Normal, 
            All
        }

        [MyBrowsable, Category("Iterations")]
        [YAXSerializableField(DefaultValue = MyLoopOperation.Normal)]
        public MyLoopOperation LoopType{ get; set; }

        private MyExecutionBlock m_oneShotTasks;

        public virtual MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            m_oneShotTasks = defaultInitPhasePlan;
            switch (LoopType)
            {
                case MyLoopOperation.All:
                {
                    return new MyExecutionBlock();
                }
                case MyLoopOperation.Normal:
                default:
                {
                    return defaultInitPhasePlan;
                }
            }
        }

        public virtual MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            switch (LoopType)
            {
                case MyLoopOperation.All:
                {
                    var res = new MyLoopBlock(
                        x => x < Iterations,
                        new MyExecutionBlock(
                            m_oneShotTasks,
                            defaultPlan
                        )
                    );
                    return res;
                }
                case MyLoopOperation.Normal:
                default:
                {
                    return new MyLoopBlock(
                        x => x < Iterations,
                        defaultPlan
                    );
                }
            }
        }
    }
}
