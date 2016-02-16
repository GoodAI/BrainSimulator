using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Testing.BrainUnit;
using Xunit;

namespace GoodAI.Modules.Tests
{
    public sealed class ReallocationBrainTest : BrainTest
    {
        private float m_difference;

        public ReallocationBrainTest()
        {
            BrainFileName = "reallocation-test.brain";

            MaxStepCount = 5;
            InspectInterval = 1;
        }

        public override bool ShouldStop(IBrainScan scan)
        {
            m_difference = scan.GetValues(22).Sum() - scan.GetValues(20).Sum();

            return IsCheckSumFailed();
        }

        public override void Check(IBrainScan b)
        {
            Assert.False(IsCheckSumFailed());
        }

        private bool IsCheckSumFailed()
        {
            return Math.Abs(m_difference) > 0;
        }
    }
}