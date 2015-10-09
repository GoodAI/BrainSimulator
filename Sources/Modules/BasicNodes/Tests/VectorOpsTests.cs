using GoodAI.BasicNodes.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace BasicNodesTests
{
    public class VectorOpsTests
    {
        public const int precision = 3;

        [Fact]
        public void DegreeToRadian()
        {
            Assert.Equal(0, VectorOps.DegreeToRadian(0));
            Assert.Equal(Math.PI, VectorOps.DegreeToRadian(180), precision);
            Assert.Equal(2 * Math.PI, VectorOps.DegreeToRadian(360), precision);
        }

        [Fact]
        public void RadianToDegree()
        {
            Assert.Equal(0, VectorOps.RadianToDegree(0));
            Assert.Equal(180 / Math.PI, VectorOps.RadianToDegree(1), precision);
            Assert.Equal(180, VectorOps.RadianToDegree((float)Math.PI));
        }
    }
}
