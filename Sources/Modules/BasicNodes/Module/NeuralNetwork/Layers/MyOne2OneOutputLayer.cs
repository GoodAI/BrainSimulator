using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YAXLib;

namespace GoodAI.Modules.NeuralNetwork.Layers
{
    /// <author>GoodAI</author>
    /// <meta>hk</meta>
    /// <status>Working</status>
    /// <summary>One-2-One Output layer node.</summary>
    /// <description>
    /// </description>
    public class MyOutputOne2OneLayer : MyAbstractOutputLayer, IMyCustomTaskFactory
    {
        // Memory blocks
        [MyInputBlock(1)]
        public override MyMemoryBlock<float> Target
        {
            get { return GetInput(1); }
        }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
          //  if (PreviousLayer != null)
          //      Neurons = PreviousLayer.Output.Count;
            if (Target != null)
                Neurons = Target.Count;

            base.UpdateMemoryBlocks();

            if (Neurons > 0)
            {
                Weights.Count = Neurons;
                Bias.Count = Neurons;

                // SGD allocations
                Delta.Count = Neurons;
                PreviousWeightDelta.Count = Weights.Count;  // momentum method
                PreviousBiasDelta.Count = Bias.Count; // momentum method

                // RMSProp allocations
                MeanSquareWeight.Count = Weights.Count;
                MeanSquareBias.Count = Bias.Count;

                // Adadelta allocation
                // AdadeltaWeight.Count = Weights.Count;
                // AdadeltaBias.Count = Bias.Count;
            }
        }

        public void CreateTasks()
        {
            ForwardTask = new MyOneToOneForwardTask();
            DeltaBackTask = new MyOneToOneDeltaBackTask();
        }

        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);
            validator.AssertError(Target != null, this, "Target not defiend");
        }
    }
}
