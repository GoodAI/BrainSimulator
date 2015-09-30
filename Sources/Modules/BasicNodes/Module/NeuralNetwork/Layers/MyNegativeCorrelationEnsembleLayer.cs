using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using GoodAI.Modules.NeuralNetwork.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;
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
    /// <meta>mbr</meta>
    /// <status>Working</status>
    /// <summary>Negative Correlation Ensemble Layer.</summary>
    /// <description>
    /// Negative Correlation Ensemble Layer takes as input the outputs from multiple output layers (all have to have the same target) and tries to specialize them.
    /// </description>
    public class MyNegativeCorrelationEnsembleLayer : MyAbstractLayer, IMyCustomTaskFactory, IMyVariableBranchViewNodeBase
    {
        [ReadOnly(false)]
        [YAXSerializableField, YAXElementFor("IO")]
        public override int InputBranches
        {
            get { return base.InputBranches; }
            set { base.InputBranches = value; }
        }

        [MyOutputBlock(1)]
        public MyMemoryBlock<float> Cost
        {
            get { return GetOutput(1); }
            set { SetOutput(1, value); }
        }

        public override ConnectionType Connection
        {
            get { return ConnectionType.ONE_TO_ONE; }
        }

        public MyNegativeCorrelationInitTask InitTask { get; protected set; }

        //Memory blocks size rules
        public override void UpdateMemoryBlocks()
        {
            // automatically set number of neurons to the same size as target
            if (GetInput(0) != null)
            {
                Neurons = GetInput(0).Count;
                Output.Count = GetInput(0).Count;
                Delta.Count = GetInput(0).Count;
            }

            base.UpdateMemoryBlocks(); // call after number of neurons are set
        }

        public void CreateTasks()
        {
            ForwardTask = new MyNegativeCorrelationForwardTask();
            DeltaBackTask = new MyNegativeCorrelationBackDeltaTask();
        }


        public override void Validate(MyValidator validator)
        {
            base.Validate(validator);

            if (validator.ValidationSucessfull)
            {
                // all inputs have equal size (so take first and check others)
                for (int i = 1, size = GetInputSize(0); i < InputBranches; i++)
                {
                    validator.AssertError(size == GetInputSize(i), this, "All inputs must be the same size");
                }
                validator.AssertError(InputBranches >= 2, this, "At least one target and one input have to be set");
            }
        }

        public override string Description
        {
            get
            {
                return "Negative correlation";
            }
        }
    }
}
