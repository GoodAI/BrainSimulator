using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Layers;
using System.ComponentModel;

namespace GoodAI.Modules.RBM.Tasks
{
    /// <summary>
    /// Initializes RBM Layer memory/parameters with zeroes where needed.
    /// </summary>
    [Description("RBMInitLayer"), MyTaskInfo(OneShot = true)]
    public class MyRBMInitLayerTask : MyTask<MyAbstractLayer>
    {
        //Properties

        //parameterless constructor
        public MyRBMInitLayerTask() { }

        //Kernel initialization
        public override void Init(int nGPU)
        {
        }
        
        //Task execution
        public override void Execute()
        {            
            //if (Owner.ParentNetwork.DefaultTraining)
            //    Owner.TrainingSignal.Raise();
            //else
            //    Owner.TrainingSignal.Keep();

            // init vars to 0
            Owner.Output.Fill(0);
           
            if (Owner is MyRBMLayer)
            {
                ((MyRBMLayer)Owner).Bias.Fill(0);
                ((MyRBMLayer)Owner).PreviousBiasDelta.Fill(0);
            }
            else if (Owner is MyRBMInputLayer)
            {
                ((MyRBMInputLayer)Owner).Bias.Fill(0);
            }
        }

    }
}
