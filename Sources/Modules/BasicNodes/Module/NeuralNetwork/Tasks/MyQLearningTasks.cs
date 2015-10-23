using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Group;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;
using System.Linq;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Tasks
{
    /// <author>GoodAI</author>
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>This is the main QLearning algorithm, that uses the current timestep as 't+1' and the previous timestep as 't'
    /// </summary>
    /// <description></description>
    [Description("QLearning"), MyTaskInfo(OneShot = false)]
    public class MyQLearningTask : MyTask<MyQLearningLayer>
    {
        // properties
        [YAXSerializableField(DefaultValue = 0.99f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float DiscountFactor { get; set; }

        [YAXSerializableField(DefaultValue = 0.0f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float BindLower { get; set; }

        [YAXSerializableField(DefaultValue = 1.00f)]
        [MyBrowsable, Category("\tHyperParameters")]
        public float BindUpper { get; set; }

        [YAXSerializableField(DefaultValue = false)]
        [MyBrowsable, Category("\tHyperParameters")]
        public bool BindTarget { get; set; }

        public MyQLearningTask() { } //parameterless constructor

        public override void Init(int nGPU)
        {

        }

        public override void Execute() //Task execution
        {
            if (SimulationStep == 0)
            {
                Owner.Output.FillAll(0);
            }


            Owner.Action.SafeCopyToHost();
            Owner.PreviousAction.SafeCopyToHost();


           // SHOULD NOT BE HERE, because Owner.Action is the action vector computed in this step, not the previous one... Owner.PreviousAction.CopyFromMemoryBlock(Owner.Action, 0, 0, Owner.Neurons);
            Owner.PreviousReward.CopyFromMemoryBlock(Owner.Reward, 0, 0, 1);
            Owner.PreviousAction.SafeCopyToHost(); // manipulate at host
            Owner.PreviousReward.SafeCopyToHost(); // manipulate at host


            // backup outputs in host memory and find the best possible value
            Owner.Output.SafeCopyToHost();


            float maxValue = Owner.Output.Host.Max();

            // copying reward to host must take place here - before network inputs backup happens - it would produce problems if in opposite order and net input and Reward being the same memblock
           // Owner.Reward.SafeCopyToHost();  //PD
            // backup network inputs in host memory
            Owner.ParentNetwork.FirstTopologicalLayer.Input.SafeCopyToHost();

            // do a forward pass with the previous inputs
            Owner.ParentNetwork.FirstTopologicalLayer.Input.CopyFromMemoryBlock(Owner.PreviousInput, 0, 0, Owner.PreviousInput.Count);
            Owner.ParentNetwork.FeedForward();

            // set the target: q(s(t-1), a) -> reward(t-1) + q(s(t), maxarg(a)) * discountfactor // TODO: consider reward(t-1) vs reward(t) :: reward(t-1) is correct, but reward(t) works better with our implementation of breakout
            Owner.Target.CopyFromMemoryBlock(Owner.Output, 0, 0, Owner.Neurons);
            Owner.Target.SafeCopyToHost(); // manipulate at host
            
            //find action which was chosen in previous step
            int chosenAction = 0;
            float maxReward = float.MinValue;
            for (int a = 0; a < Owner.Neurons; a++)
            {
                if (Owner.PreviousAction.Host[a] > maxReward)
                {
                    chosenAction = a;
                    maxReward = Owner.PreviousAction.Host[a];
                }
                
            }

            //float normalize = 0;
            //for (int a = 0; a < Owner.Neurons; a++)
            //    normalize += Owner.PreviousAction.Host[a];
            //if (normalize > 0)
            //{
            //    for (int a = 0; a < Owner.Neurons; a++)
            //        Owner.PreviousAction.Host[a] /= normalize;
            //}
            //else
            //{
            //    for (int a = 0; a < Owner.Neurons; a++)
            //        Owner.PreviousAction.Host[a] = 1.0f / Owner.Neurons;
            //}

                                          //prev reward + discount factor * expected reward
            float target = (Owner.PreviousReward.Host[0] + DiscountFactor * maxValue);  
            if (BindTarget)
            {
                if (target > BindUpper)
                    target = BindUpper;
                else if (target < BindLower)
                    target = BindLower;
            }
            Owner.Target.Host[chosenAction] = target;

            

            //for (int a = 0; a < Owner.Neurons; a++)
            //{
            //    float target;
            //    //float target = Owner.PreviousAction.Host[a] * (value * DiscountFactor) + (1 - Owner.PreviousAction.Host[a]) * Owner.Target.Host[a];
            //    target = Owner.PreviousAction.Host[a] * (Owner.Reward.Host[0] + maxValue * DiscountFactor) + (1 - Owner.PreviousAction.Host[a]) * Owner.Target.Host[a];
            //    MyLog.DEBUG.WriteLine("target: " + target);

            //    //  target = (Owner.Reward.Host[0] + maxValue * DiscountFactor) + alpha * Owner.Target.Host[a];

            //    if (BindTarget)
            //    {
            //        if (target > BindUpper)
            //            target = BindUpper;
            //        else if (target < BindLower)
            //            target = BindLower;
            //    }
            //    Owner.Target.Host[a] = target;
            //}

            Owner.Target.SafeCopyToDevice(); // back to device

            // copy reward to previous value
           // Owner.PreviousReward.CopyFromMemoryBlock(Owner.Reward, 0, 0, 1);

            // copy action to previous value
            Owner.PreviousAction.CopyFromMemoryBlock(Owner.Action, 0, 0, Owner.Neurons);

            Owner.ParentNetwork.GetError();
            Owner.ParentNetwork.GetActiveBackpropTask().Execute(Owner);
            ((MyRMSTask)Owner.ParentNetwork.GetActiveBackpropTask()).Execute((MyAbstractWeightLayer)Owner.ParentNetwork.FirstTopologicalLayer);
          //  ((MySGDTask)Owner.ParentNetwork.GetActiveBackpropTask()).Execute((MyAbstractWeightLayer)Owner.ParentNetwork.FirstTopologicalLayer);
         //    ((MyRMSTask)Owner.ParentNetwork.GetActiveBackpropTask()).Execute((GoodAI.Modules.LSTM.MyLSTMLayer)Owner.ParentNetwork.FirstTopologicalLayer);

          // ((GoodAI.Modules.LSTM.MyLSTMLayer)Owner.ParentNetwork.FirstTopologicalLayer).
            //GoodAI.Modules.LSTM.Tasks.MyLSTMPartialDerivativesTask t;
            //t.Execute();

         //   Owner.ParentNetwork.GetActiveBackpropTask().Execute(Owner.ParentNetwork.FirstTopologicalLayer);
          //  ((GoodAI.Modules.LSTM.MyLSTMLayer)Owner.ParentNetwork.FirstTopologicalLayer)
          //      MyLSTMUpdateWeightsTask().


            // restore output values from host
            Owner.Output.SafeCopyToDevice();

            // restore input values from host
            Owner.ParentNetwork.FirstTopologicalLayer.Input.SafeCopyToDevice();
            Owner.PreviousInput.CopyFromMemoryBlock(Owner.ParentNetwork.FirstTopologicalLayer.Input, 0, 0, Owner.PreviousInput.Count);

        }
    }
}