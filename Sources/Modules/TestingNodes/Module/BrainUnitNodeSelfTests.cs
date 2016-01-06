using GoodAI.Testing.BrainUnit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.TestingNodes
{
    public sealed class BrainTestCanBeDiscovered : BrainTest
    {
        public BrainTestCanBeDiscovered()
        {
            BrainFileName = "brain-unit-node-self-test.brain";
            MaxStepCount = 1;
        }

        public override void Check(IBrainScan b)
        {
            // pass
        }

    }
}
