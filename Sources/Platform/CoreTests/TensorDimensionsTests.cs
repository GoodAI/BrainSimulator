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
            var dims = new TensorDimensionsV1(2, 3);

            Assert.Equal(2, dims.Count);
            Assert.Equal(2, dims[0]);
            Assert.Equal(3, dims[1]);
            Assert.False(dims.IsCustom);  // dimensions created in code are "default" and should not be saved to project
        }

        [Fact]
        public void CanBeComputedEvaluatesToTrue()
        {
            var dims = new TensorDimensionsV1(3, 4, -1, 5) { Size = 3*4*5*13 };

            Assert.True(dims.CanBeComputed);
            Assert.Equal(13, dims[2]);  // also check that the free dimension was correctly computed
        }

        [Fact]
        public void CanBeComputedEvaluatesToFalse()
        {
            var dims = new TensorDimensionsV1(3, 4, -1, 5) { Size = 37 };

            Assert.False(dims.CanBeComputed);
        }

        [Fact]
        public void EmptyDimensionsCanBeComputed()
        {
            var dims = new TensorDimensionsV1();
            Assert.False(dims.CanBeComputed);

            dims.Size = 4;
            Assert.True(dims.CanBeComputed);
            Assert.Equal(4, dims[0]);
        }

        [Fact]
        public void ComputedDimCanBeOne()
        {
            var dims = new TensorDimensionsV1(-1, 10) { Size = 10 };

            Assert.True(dims.CanBeComputed);
            Assert.Equal(1, dims[0]);
        }

        [Fact]
        public void DimensionsOfSizeOneAreAllowed()
        {
            var dims = new TensorDimensionsV1();

            dims.Set(new []{ 5, 1, 1 });
        }

        [Fact]
        public void ParseKeepsDimensionsOfSizeOne()
        {
            var dims = new TensorDimensionsV1();

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
            var dims = new TensorDimensionsV1(3, 3) { Size = 4 };

            Assert.Equal("3×3 (!)", dims.Print());
        }

        [Fact]
        public void DoesNotPrintTrailingOnes()
        {
            var dims = new TensorDimensionsV1(5, 1, 1) { Size = 5 };

            Assert.Equal("5", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void PrintsComputedTrailingOne()
        {
            var dims = new TensorDimensionsV1(4, 2, -1) { Size = 8 };

            Assert.Equal("4×2×1", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void PrintsOneOne()
        {
            var dims = new TensorDimensionsV1(1, 1);

            Assert.Equal("1 (!)", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void PrintsLeadingOrMiddleOnes()
        {
            var dims = new TensorDimensionsV1(1, 1, -1, 5, 1, 2, 1);

            Assert.Equal("1×1×?×5×1×2", dims.Print(hideTrailingOnes: true));
        }

        [Fact]
        public void ParseAutoAddsLeadingDim()
        {
            var dims = new TensorDimensionsV1();
            dims.Parse("2, 2, 2");

            Assert.Equal(4, dims.Count);
            Assert.Equal(-1, dims[0]);
            Assert.Equal(2, dims[1]);
        }

        [Fact]
        public void ParseDoesNotAutoAddDimWhenSizeMatches()
        {
            var dims = new TensorDimensionsV1() { Size = 2*2*2 };
            dims.Parse("2, 2, 2");

            Assert.Equal(3, dims.Count);
            Assert.Equal(2, dims[0]);
            Assert.Equal(2, dims[1]);
        }

        [Fact]
        public void SizeGetsUpdatedWhenDimsAssignedToMemBlock()
        {
            var memBlock = new MyMemoryBlock<float>
            {
                Count = 10,
                Dims = new TensorDimensionsV1(2)
            };

            Assert.Equal(memBlock.Count, memBlock.Dims.Size);
        }

        private static MyMemoryBlock<float> GetMemBlockWithCustomDims(string dimensionsSource)
        {
            var customDims = new TensorDimensionsV1();
            customDims.Parse(dimensionsSource);

            return new MyMemoryBlock<float> { Dims = customDims };
        }

        [Fact]
        public void CodeGeneratedDimsDoNotOverwriteCustomOnes()
        {
            MyMemoryBlock<float> memBlock = GetMemBlockWithCustomDims("2, -1, 2");

            memBlock.Dims = new TensorDimensionsV1(33);  // this assignment should be ignored

            Assert.Equal(3, memBlock.Dims.Count);
            Assert.Equal(2, memBlock.Dims[0]);
        }

        [Fact]
        public void CustomDimsDoOverwritePreviousOnes()
        {
            MyMemoryBlock<float> memBlock = GetMemBlockWithCustomDims("2, -1, 2");

            memBlock.Dims = new TensorDimensionsV1(33) { IsCustom = true };  // this assignment must NOT be ignored

            Assert.Equal(1, memBlock.Dims.Count);
            Assert.Equal(33, memBlock.Dims[0]);
        }

        private TensorDimensions m_defaultDims = new TensorDimensions(5, 3, 2);

        [Fact]
        public void RankReturnsNumberOfDims()
        {
            Assert.Equal(3, m_defaultDims.Rank);
        }

        [Fact]
        public void PrintsEmptyDims()
        {
            var dims = new TensorDimensions();

            Assert.Equal("0", dims.Print());
        }

        [Fact]
        public void PrintsDims()
        {
            Assert.Equal("5×3×2", m_defaultDims.Print());
            Assert.Equal("5×3×2 [30]", m_defaultDims.Print(printTotalSize: true));
        }

    }
}
