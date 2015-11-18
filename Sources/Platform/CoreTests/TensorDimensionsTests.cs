using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using Xunit;

namespace CoreTests
{
    public class TensorDimensionsTests
    {
        [Fact]
        public void ConstructsWithVariableNumberOfParams()
        {
            var dims = new TensorDimensions(2, 3);

            Assert.Equal(2, dims.Count);
            Assert.Equal(2, dims[0]);
            Assert.Equal(3, dims[1]);
        }

        [Fact]
        public void CanBeComputedEvaluatesToTrue()
        {
            var dims = new TensorDimensions(3, 4, -1, 5);
            dims.Size = 3 * 4 * 5 * 13;

            Assert.True(dims.CanBeComputed);
            Assert.Equal(13, dims[2]);  // also check that the free dimension was correctly computed
        }

        [Fact]
        public void CanBeComputedEvaluatesToFalse()
        {
            var dims = new TensorDimensions(3, 4, -1, 5);
            dims.Size = 37;

            Assert.False(dims.CanBeComputed);
        }

        [Fact]
        public void EmptyDimensionsCanBeComputed()
        {
            var dims = new TensorDimensions();
            Assert.False(dims.CanBeComputed);

            dims.Size = 4;
            Assert.True(dims.CanBeComputed);
            Assert.Equal(4, dims[0]);
        }
    }
}
