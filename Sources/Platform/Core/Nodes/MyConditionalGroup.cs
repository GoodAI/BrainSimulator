using GoodAI.Core.Execution;
using GoodAI.Core.Signals;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using System;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Core.Nodes
{
    /// <author>GoodAI</author>
    /// <meta>df</meta>
    /// <status>Working</status>
    /// <summary>Groups several nodes to one entity. Inside of a group is executed only on specific incoming signal.</summary>
    /// <description>Enables nodes to be put inside a group, which makes model more structured. 
    /// Nodes inside are executed only when appropriate signal is present.</description>
    public class MyConditionalGroup : MyNodeGroup, IMyCustomExecutionPlanner
    {
        public MyProxySignal ActiveSignal { get; set; }

        [MyBrowsable, TypeConverter(typeof(MySignal.MySignalTypeConverter))]
        [YAXSerializableField(DefaultValue = "<none>")]
        public string Signal { get; set; }

        public override string Description
        {
            get
            {
                return "Active on:";
            }
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
        }

        public override void UpdateMemoryBlocks()
        {
            base.UpdateMemoryBlocks();

            if (!Signal.Equals("<none>"))
            {
                ActiveSignal.Source = MySignal.CreateSignalByDefaultName(Signal);

                if (ActiveSignal.Source != null)
                {
                    ActiveSignal.Source.Owner = this;
                    ActiveSignal.Source.Name = MyProject.RemovePostfix(ActiveSignal.Source.DefaultName, "Signal");
                }
            }
            else
            {
                ActiveSignal.Source = null;
            }
        }

        internal bool IsActive()
        {            
            return (ActiveSignal.Source != null && ActiveSignal.IsIncomingRised());                         
        }
        
        public MyExecutionBlock CreateCustomExecutionPlan(MyExecutionBlock defaultPlan)
        {
            //get rid of signal tasks
            IMyExecutable[] content = new IMyExecutable[defaultPlan.Children.Length - 2];
            Array.Copy(defaultPlan.Children, 1, content, 0, defaultPlan.Children.Length - 2); 

            //place if block inside
            return new MyExecutionBlock(
                defaultPlan.Children[0],
                new MyIfBlock(IsActive, content),
                defaultPlan.Children.Last()) { Name = defaultPlan.Name };
        }

        public MyExecutionBlock CreateCustomInitPhasePlan(MyExecutionBlock defaultInitPhasePlan)
        {
            return defaultInitPhasePlan;
        }
    }
}
