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
    /// <status>Working</status>
    /// <summary>
    /// Group that runs all Nodes inside multiple times per one simulation step. 
    /// </summary>
    /// <description>
    /// Run all Nodes inside multiple times per step.
    /// <h3>Parameters</h3>
    /// <ul>
    ///     <li> <b>Iterations:</b> How many Iterations to run everything inside at one SimulationStep.</li>
    ///     <li> <b>LoopType:</b> "Normal" mode will use the default plan, "All" means that LoopGroup will run OneShot tasks too.</li>
    /// </ul>
    /// </description>
    public class MyLoopGroup : MyNodeGroup, IMyCustomExecutionPlanner
    {
        [YAXSerializableField(DefaultValue = 3)]
        [MyBrowsable, Category("Iterations"), Description("How many times per simulation step to loop the group")]
        public int Iterations { get; set; }

        public enum MyLoopOperation
        {
            Normal, 
            All
        }

        [MyBrowsable, Category("Iterations"), Description("Specifies mode of looping. Normal ignores OneShot tasks while ALL loops all tasks including OneShot tasks")]
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
