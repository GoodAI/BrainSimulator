using GoodAI.Testing.BrainUnit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Xunit;

namespace GoodAI.Modules.Tests
{
    public sealed class AccumulatorCanCountToTen : BrainTest
    {
        public AccumulatorCanCountToTen()
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

    public abstract class AccumulatorTestBase : BrainTest
    {
        protected AccumulatorTestBase()
        {
            BrainFileName = "accumulator-test.brain";

            MaxStepCount = 10;
        }
    }

    public sealed class StopConditionWorksOnAccumulator : AccumulatorTestBase
    {
        public StopConditionWorksOnAccumulator()
        {
            InspectInterval = 3;
        }

        public override bool ShouldStop(IBrainScan b)
        {
            Check(b);
            return true;  // stop if Check() does not throw an exception
        }

        public override void Check(IBrainScan b)
        {
            Assert.True(b.GetValues(7)[0] > 5);
        }
    }

    public sealed class MyFailingAccumulatorTest : AccumulatorTestBase
    {
        public MyFailingAccumulatorTest()
        {
            InspectInterval = 3;
            ExpectedToFail = true;
        }

        public override void Check(IBrainScan b)
        {
            Assert.Equal(10, b.GetValues(7)[0] + 1000);  // fails, + 1000 is an inserted bug
        }
    }
}
