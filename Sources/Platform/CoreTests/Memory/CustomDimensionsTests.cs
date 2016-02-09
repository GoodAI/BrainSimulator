using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

using GoodAI.Core.Memory;

namespace CoreTests.Memory
{
    public class CustomDimensionsTests
    {
        [Fact]
        public void FactoryParsesSimpleDimensions()
        {
            var dimensionsHint = new CustomDimensionsHint("2, 3, 5");

            Assert.Equal(3, dimensionsHint.Rank);
            Assert.Equal(2, dimensionsHint[0]);
            Assert.Equal(3, dimensionsHint[1]);
            Assert.Equal(5, dimensionsHint[2]);
        }
        // TODO(Premek): Test more cases such as with *s, also test checking of invalid cases.
        // TODO: Test IsFullyDefined

        [Fact]
        public void ElementCountWorksForWildcardHints()
        {
            Assert.Equal(6, new CustomDimensionsHint("2, *, 3").ElementCount);
        }

        [Fact]
        public void CustomDimensionsAreSameAsHintWhenCountMatches()
        {
            TensorDimensions dims = new CustomDimensionsHint("2, 3, 5").ComputeDimensions(30);

            Assert.Equal(new TensorDimensions(2, 3, 5), dims);
        }

        [Fact]
        public void WildcardDimensionIsCorrectlyComputed()
        {
            TensorDimensions dims = new CustomDimensionsHint("2, *, 3").ComputeDimensions(30);

            Assert.Equal(dims, new TensorDimensions(2, 5, 3));
        }

        [Fact]
        public void FallsBackToOneDimWhenCountDoesNotMatch()
        {
            // TODO: Test that the fallback is logged.
            TensorDimensions dims = new CustomDimensionsHint("2, 3, 5").ComputeDimensions(47);

            Assert.Equal(new TensorDimensions(47), dims);
        }

        [Fact]
        public void FallsBackToOneDimWhenCountIsNotDivisible()
        {
            // TODO: Test that the fallback is logged.
            TensorDimensions dims = new CustomDimensionsHint("2, *, 3").ComputeDimensions(47);

            Assert.Equal(new TensorDimensions(47), dims);
        }

        [Fact]
        public void FallsBackToOneDimWhenEmpty()
        {
            Assert.Equal(new TensorDimensions(47), CustomDimensionsHint.Empty.ComputeDimensions(47));
        }
    }
}
