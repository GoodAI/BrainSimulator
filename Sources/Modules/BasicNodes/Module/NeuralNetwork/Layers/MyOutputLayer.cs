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
    /// <meta>ph</meta>
    /// <status>Working</status>
    /// <summary>Output layer node.</summary>
    /// <description>
    /// The output layer takes a target as input, and automatically scales it's neurons to fit the target.<br></br>
    /// The cost can be observed or manipulated as an output
    /// </description>
    public class MyOutputLayer : MyAbstractOutputLayer
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
            // automatically set number of neurons to the same size as target
            if (Target != null)
                Neurons = Target.Count;

            base.UpdateMemoryBlocks(); // call after number of neurons are set
        }
    }
}
