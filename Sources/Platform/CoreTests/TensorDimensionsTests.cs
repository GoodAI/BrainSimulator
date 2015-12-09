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
            Assert.False(dims.IsCustom);  // dimensions created in code are "default" and should not be saved to project
        }

        [Fact]
        public void CanBeComputedEvaluatesToTrue()
        {
            var dims = new TensorDimensions(3, 4, -1, 5) { Size = 3*4*5*13 };

            Assert.True(dims.CanBeComputed);
            Assert.Equal(13, dims[2]);  // also check that the free dimension was correctly computed
        }

        [Fact]
        public void CanBeComputedEvaluatesToFalse()
        {
            var dims = new TensorDimensions(3, 4, -1, 5) { Size = 37 };

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

        [Fact]
        public void ComputedDimCanBeOne()
        {
            var dims = new TensorDimensions(-1, 10) { Size = 10 };

            Assert.True(dims.CanBeComputed);
            Assert.Equal(1, dims[0]);
        }

        [Fact]
        public void DimensionsOfSizeOneNotAllowed()
        {
            var dims = new TensorDimensions();

            Assert.Throws<InvalidDimensionsException>(() => dims.Set(new []{ 5, 1, 1 }));
        }

        [Fact]
        public void ParseSkipsDimensionsOfSizeOne()
        {
            var dims = new TensorDimensions();

            dims.Parse("1, 5, 1, 1");

            Assert.Equal(2, dims.Count);  // *, 5
            Assert.Equal(5, dims[1]);
            Assert.NotEqual("", dims.LastSetWarning);
        }

        [Fact]
        public void SizeGetsUpdatedWhenDimsAssignedToMemBlock()
        {
            var memBlock = new MyMemoryBlock<float>
            {
                Count = 10,
                Dims = new TensorDimensions(2)
            };

            Assert.Equal(memBlock.Count, memBlock.Dims.Size);
        }

        private static MyMemoryBlock<float> GetMemBlockWithCustomDims()
        {
            var customDims = new TensorDimensions();
            customDims.Parse("2, -1, 2");

            return new MyMemoryBlock<float> {Dims = customDims};
        }

        [Fact]
        public void CodeGeneratedDimsDoNotOverwriteCustomOnes()
        {
            MyMemoryBlock<float> memBlock = GetMemBlockWithCustomDims();

            memBlock.Dims = new TensorDimensions(33);  // this assignment should be ignored

            Assert.Equal(3, memBlock.Dims.Count);
            Assert.Equal(2, memBlock.Dims[0]);
        }

        [Fact]
        public void CustomDimsDoOverwritePreviousOnes()
        {
            MyMemoryBlock<float> memBlock = GetMemBlockWithCustomDims();

            memBlock.Dims = new TensorDimensions(33) { IsCustom = true };  // this assignment must NOT be ignored

            Assert.Equal(1, memBlock.Dims.Count);
            Assert.Equal(33, memBlock.Dims[0]);
        }
    }
}
