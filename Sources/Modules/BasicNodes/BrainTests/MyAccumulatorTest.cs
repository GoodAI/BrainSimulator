using GoodAI.Testing.BrainUnit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GoodAI.Modules.Tests
{
    public class MyAccumulatorTest : BrainTest
    {
        public MyAccumulatorTest()
        {
            BrainFileName = "../../../BrainTests/Brains/accumulator-test.brain";

            MaxStepCount = 10;
            InspectInterval = 2;
        }
    }
}
