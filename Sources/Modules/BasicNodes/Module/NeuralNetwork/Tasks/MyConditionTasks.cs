using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    // checks the if node is ready for learning or not depending on any of two things
    // 1) CanLearn input flag (if it is not null)
    // 2) Incoming IsLearning signal
    [Description("Condition"), MyTaskInfo(OneShot = false, Disabled = false)]
    public class MyNodeConditionTask : MyTask<MyAbstractLayer>
    {
        public override void Init(int nGPU)
        {
            if (Owner.CanLearn != null)
                Owner.CanLearn.SafeCopyToHost();
        }

        public override void Execute()
        {
            // all nodes' (including output nodes') learning can be turned off by CanLearn = 0 (false)
            bool canLearn = true;
            if (Owner.CanLearn != null)
                canLearn = System.Convert.ToBoolean(Owner.CanLearn.Host[0]);

            // if incoming learning signal is rised or has been overriden by canLearn flag
            if (canLearn && Owner.IsLearning.IsIncomingRised())
                Owner.IsLearning.Raise();
            else
                Owner.IsLearning.Drop();

            // enable or disable learning tasks according to IsLearning signal
            if (Owner.IsLearning.IsRised())
                // should be overloaded for nodes with non-standard set of learning tasks
                Owner.EnableLearningTasks();
            else
                // should be overloaded for nodes with non-standard set of learning tasks
                Owner.DisableLearningTasks();
        }
    }
    
    // checks the if inner barrier condition is satisfied
    // 1) If inner conditions are satisfied (only Output nodes)
    [Description("Condition"), MyTaskInfo(OneShot = false, Disabled = false)]
    public class MyBarrierConditionTask : MyTask<MyAbstractOutputLayer>
    {
        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("\tNumberOfStepsCondition"), ReadOnly(true)]
        public virtual int LearningStepCounter { get; set; }

        [YAXSerializableField(DefaultValue = 0)]
        [MyBrowsable, Category("\tNumberOfStepsCondition")]
        public virtual int NumberOfLearningSteps { get; set; }

        public override void Init(int nGPU)
        {
            LearningStepCounter = 0;

            Owner.IsLearned.SafeCopyToHost();
            Owner.IsLearned.Host[0] = Convert.ToSingle(false);
            Owner.IsLearned.SafeCopyToDevice();
        }

        public override void Execute()
        {
            bool isLearned = Convert.ToBoolean(Owner.IsLearned.Host[0]);

            // all nodes' (including output nodes') learning can be turned off by CanLearn = 0 (false)
            bool canLearn = true;
            if (Owner.CanLearn != null)
                canLearn = System.Convert.ToBoolean(Owner.CanLearn.Host[0]);

            if (canLearn && Owner.IsLearning.IsIncomingRised() && !isLearned)
                Owner.IsLearning.Raise();
            else
                Owner.IsLearning.Drop();

            // output layers have inner conditions (such as number of steps)
            // which enables learning when satisfied
            if (0 < NumberOfLearningSteps && NumberOfLearningSteps <= LearningStepCounter)
            {
                bool prevIsLearned = Convert.ToBoolean(Owner.IsLearned.Host[0]);
                Owner.IsLearned.Host[0] = Convert.ToSingle(true);
                if (prevIsLearned == false) // if changed, copy to device
                    Owner.IsLearned.SafeCopyToDevice();
            }
            else
            {
                bool prevIsLearned = Convert.ToBoolean(Owner.IsLearned.Host[0]);
                Owner.IsLearned.Host[0] = Convert.ToSingle(false);
                if (prevIsLearned == true) // if changed, copy to device
                    Owner.IsLearned.SafeCopyToDevice();
            }

            // enable or disable learning tasks according to IsLearning signal
            if (Owner.IsLearning.IsRised())
            {
                // should be overloaded for nodes with non-standard set of learning tasks
                Owner.EnableLearningTasks();
                //Owner.IsLearning.Raise();
                LearningStepCounter++;
            }
            else
            {
                // should be overloaded for nodes with non-standard set of learning tasks
                Owner.DisableLearningTasks();
            }
        }
    }
}
