using GoodAI.Testing.BrainUnit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Xunit;

namespace GoodAI.Modules.Tests
{
    public sealed class MyAccumulatorTest : BrainTest
    {
        public MyAccumulatorTest()
        {
            BrainFileName = "accumulator-test.brain";

            MaxStepCount = 10;
            InspectInterval = 2;
        }

        public override bool ShouldStop(IBrainScan b)
        {
            return b.GetValues(7)[0] > 15;
        }

        public override void Check(IBrainScan b)
        {
            Assert.Equal(10, b.GetValues(7)[0]);
        }
    }

    public sealed class MyFailingAccumulatorTest : BrainTest
    {
        public MyFailingAccumulatorTest()
        {
            BrainFileName = "accumulator-test.brain";

            MaxStepCount = 10;
            InspectInterval = 3;
        }

        public override bool ShouldStop(IBrainScan b)
        {
            return b.GetValues(7)[0] > 5;
        }

        public override void Check(IBrainScan b)
        {
            Assert.Equal(10, b.GetValues(7)[0]);
        }
    }
}
