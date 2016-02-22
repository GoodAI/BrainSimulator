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
        public void ParsesSimpleDimensions()
        {
            var dimensionsHint = CustomDimensionsHint.Parse("2, 3, 5");

            Assert.Equal(new CustomDimensionsHint(2, 3, 5), dimensionsHint);
        }

        // ReSharper disable once MemberCanBePrivate.Global
        public static readonly TheoryData<string, CustomDimensionsHint, bool, bool> ValidParseData
            = new TheoryData<string, CustomDimensionsHint, bool, bool>
            {
                {"1", new CustomDimensionsHint(1), /* IsEmpty: */ false, /* IsFullyDefined: */ true},
                {"3, 100, *", new CustomDimensionsHint(3, 100, -1), false, false },
                {"*", new CustomDimensionsHint(-1), false, false },
                {"", CustomDimensionsHint.Empty, true, false }  // Empty is *not* fully defined by definition (OK?)
            };
        
        [Theory, MemberData("ValidParseData")]
        public void ParsingAndFlaggingTheory(
            string source, CustomDimensionsHint expectedHint, bool isEmpty, bool isFullyDefined)
        {
            var hint = CustomDimensionsHint.Parse(source);

            Assert.True(hint.Equals(expectedHint));
            Assert.Equal(isEmpty, hint.IsEmpty);
            Assert.Equal(isFullyDefined, hint.IsFullyDefined);
        }

        [Theory]
        [InlineData("foo")]
        [InlineData("*, 2, *")]
        [InlineData("*, 2, bar, *")]
        public void ParsingErrorsTheory(string source)
        {
            Assert.Throws<InvalidDimensionsException>(() => { var hint = CustomDimensionsHint.Parse(source); });
        }

        [Fact]
        public void ElementCountWorksForWildcardHints()
        {
            Assert.Equal(6, CustomDimensionsHint.Parse("2, *, 3").ElementCount);
        }

        [Fact]
        public void CustomDimensionsAreSameAsHintWhenCountMatches()
        {
            TensorDimensions dims;
            bool didApply = CustomDimensionsHint.Parse("2, 3, 5").TryToApply(new TensorDimensions(30), out dims);

            Assert.Equal(new TensorDimensions(2, 3, 5), dims);
            Assert.True(didApply);
        }

        [Fact]
        public void WildcardDimensionIsCorrectlyComputed()
        {
            TensorDimensions dims = CustomDimensionsHint.Parse("2, *, 3").TryToApply(new TensorDimensions(30));

            Assert.Equal(dims, new TensorDimensions(2, 5, 3));
        }

        // ReSharper disable once MemberCanBePrivate.Global
        public static readonly TheoryData<CustomDimensionsHint, TensorDimensions> ApplyFallbackData
            = new TheoryData<CustomDimensionsHint, TensorDimensions>
            {
                // TryToApply falls back to original dimensions ...
                // when count does not match
                { CustomDimensionsHint.Parse("2, 3, 5"), new TensorDimensions(7, 13) },

                // when count is not divisible (to be able to figure out the computed dimension)
                { CustomDimensionsHint.Parse("2, *, 5"), new TensorDimensions(7, 13) },

                // when count is divisible, but there is no computed dimension
                { CustomDimensionsHint.Parse("2"), new TensorDimensions(8) },

                // when hint is empty
                { CustomDimensionsHint.Empty, new TensorDimensions(7, 13) }
            };

        [Theory, MemberData("ApplyFallbackData")]
        public void ApplyFallbackTheory(CustomDimensionsHint hint, TensorDimensions originalDims)
        {
            TensorDimensions resultDims;
            
            bool didApply = hint.TryToApply(originalDims, out resultDims);

            Assert.Equal(originalDims, resultDims);
            Assert.False(didApply);
        }
    }
}
