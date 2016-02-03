using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using Xunit;
using Xunit.Abstractions;

namespace CoreTests
{
    public class OldTensorDimensionsTests
    {
        [Fact]
        public void ConstructsWithVariableNumberOfParams()
        {
            var dims = new OldTensorDimensions(2, 3);

            Assert.Equal(2, dims.Count);
            Assert.Equal(2, dims[0]);
            Assert.Equal(3, dims[1]);
            Assert.False(dims.IsCustom);  // dimensions created in code are "default" and should not be saved to project
        }

        [Fact]
        public void CanBeComputedEvaluatesToTrue()
        {
            var dims = new OldTensorDimensions(3, 4, -1, 5) { Size = 3 * 4 * 5 * 13 };

            Assert.True(dims.CanBeComputed);
            Assert.Equal(13, dims[2]);  // also check that the free dimension was correctly computed
        }

        [Fact]
        public void CanBeComputedEvaluatesToFalse()
        {
            var dims = new OldTensorDimensions(3, 4, -1, 5) { Size = 37 };

            Assert.False(dims.CanBeComputed);
        }

        [Fact]
        public void EmptyDimensionsCanBeComputed()
        {
            var dims = new OldTensorDimensions();
            Assert.False(dims.CanBeComputed);

            dims.Size = 4;
            Assert.True(dims.CanBeComputed);
            Assert.Equal(4, dims[0]);
        }

        [Fact]
        public void ComputedDimCanBeOne()
        {
            var dims = new OldTensorDimensions(-1, 10) { Size = 10 };

            Assert.True(dims.CanBeComputed);
            Assert.Equal(1, dims[0]);
        }

        [Fact]
        public void DimensionsOfSizeOneAreAllowed()
        {
            var dims = new OldTensorDimensions();

            dims.Set(new[] { 5, 1, 1 });
        }

        [Fact]
        public void ParseKeepsDimensionsOfSizeOne()
        {
            var dims = new OldTensorDimensions();

            dims.Parse("1, 5, *, 1, 1");

            Assert.Equal(5, dims.Count);
            Assert.Equal(1, dims[0]);
            Assert.Equal(5, dims[1]);
            Assert.Equal(1, dims[4]);
            Assert.Equal("", dims.LastSetWarning);
        }

        [Fact]
        public void PrintIndicatesMismatchedDimsAndSize()
        {
            var dims = new OldTensorDimensions(3, 3) { Size = 4 };

            Assert.Equal("3×3 (!)", dims.Print());
        }

        [Fact]
        public void DoesNotPrintTrailingOnes()
        {
            var dims = new OldTensorDimensions(5, 1, 1) { Size = 5 };

            Assert.Equal("5", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void PrintsComputedTrailingOne()
        {
            var dims = new OldTensorDimensions(4, 2, -1) { Size = 8 };

            Assert.Equal("4×2×1", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void PrintsOneOne()
        {
            var dims = new OldTensorDimensions(1, 1);

            Assert.Equal("1 (!)", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void PrintsLeadingOrMiddleOnes()
        {
            var dims = new OldTensorDimensions(1, 1, -1, 5, 1, 2, 1);

            Assert.Equal("1×1×?×5×1×2", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void ParseAutoAddsLeadingDim()
        {
            var dims = new OldTensorDimensions();
            dims.Parse("2, 2, 2");

            Assert.Equal(4, dims.Count);
            Assert.Equal(-1, dims[0]);
            Assert.Equal(2, dims[1]);
        }

        [Fact]
        public void ParseDoesNotAutoAddDimWhenSizeMatches()
        {
            var dims = new OldTensorDimensions() { Size = 2 * 2 * 2 };
            dims.Parse("2, 2, 2");

            Assert.Equal(3, dims.Count);
            Assert.Equal(2, dims[0]);
            Assert.Equal(2, dims[1]);
        }
    }
}
