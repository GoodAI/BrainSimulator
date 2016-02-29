using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using Xunit;
using Xunit.Abstractions;

namespace CoreTests.Memory
{
    public class TensorDimensionsTests : CoreTestBase
    {
        private readonly ITestOutputHelper m_output;

        public TensorDimensionsTests(ITestOutputHelper output)
        {
            m_output = output;
        }

        [Fact]
        public void ElementCountIsCachedProperly()
        {
            var dims = new TensorDimensions(3, 5, 7);

            Assert.Equal(3*5*7, dims.ElementCount);
            Assert.Equal(3*5*7, dims.ElementCount);
        }

        [Fact]
        public void HashCodeIsCachedProperly()
        {
            var dims = new TensorDimensions(3, 5, 7);

            int hashCode = dims.GetHashCode();
            Assert.Equal(hashCode, dims.GetHashCode());
        }

        [Fact]
        public void EqualsTests()
        {
            Assert.True((new TensorDimensions(2, 3, 5)).Equals(new TensorDimensions(2, 3, 5)));
            Assert.True((new TensorDimensions()).Equals(new TensorDimensions()));

            Assert.False((new TensorDimensions(2, 3, 5)).Equals(new TensorDimensions(2, 3)));
            Assert.False((new TensorDimensions(2, 3, 5)).Equals(new TensorDimensions(2, 3, 11)));

            // ReSharper disable once SuspiciousTypeConversion.Global
            Assert.False((new TensorDimensions(2, 3, 5)).Equals(0));  // Compare with some other value type.
        }

        [Fact]
        public void DefaultDimIsRankOneOfSizeZero()
        {
            TensorDimensions emptyDims = TensorDimensions.Empty;

            Assert.Equal(1, emptyDims.Rank);
            Assert.Equal(0, emptyDims[0]);
            Assert.Equal(0, emptyDims.ElementCount);
        }

        [Fact]
        public void AnyDimensionCanBeZero()
        {
            var rank1Dims = new TensorDimensions(0);
            Assert.Equal(0, rank1Dims.ElementCount);

            var rankNDims = new TensorDimensions(3, 0, 5);
            Assert.Equal(3, rankNDims[0]);
            Assert.Equal(0, rankNDims.ElementCount);
        }

        [Fact]
        public void DefaultConstructorReturnsEmptyDims()
        {
            Assert.True(TensorDimensions.Empty.Equals(new TensorDimensions()));
        }

        private readonly TensorDimensions m_testDims = new TensorDimensions(5, 3, 2);

        [Fact]
        public void RankReturnsNumberOfDims()
        {
            Assert.Equal(3, m_testDims.Rank);
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
            Assert.Equal("5×3×2", m_testDims.Print());
            Assert.Equal("5×3×2 [30]", m_testDims.Print(printTotalSize: true));
        }

        [Fact]
        public void ComputesCompatibleTensorDims()
        {
            var dims = TensorDimensions.GetBackwardCompatibleDims(10, 2);

            Assert.Equal(2, dims[0]);
            Assert.Equal(5, dims[1]);
        }

        [Fact]
        public void ComputesCompatibleTensorDimsWithWrongColumnHint()
        {
            var dims = TensorDimensions.GetBackwardCompatibleDims(10, 3);

            Assert.Equal(1, dims.Rank);
            Assert.Equal(10, dims[0]);
        }

        [Fact]
        public void ComputesCompatibleTensorDimsWithInvalidData()
        {
            var dims = TensorDimensions.GetBackwardCompatibleDims(0, 0);

            Assert.Equal(1, dims.Rank);
            Assert.Equal(0, dims[0]);
        }

        private static MyAbstractMemoryBlock GetMemBlock(TensorDimensions dims)
        {
            return new MyMemoryBlock<float> { Dims = dims };
        }

        [Fact]
        public void ColumnHintUsedWhenDivisible()
        {
            MyAbstractMemoryBlock memBlock = GetMemBlock(new TensorDimensions(12));
            memBlock.ColumnHint = 3;

            Assert.Equal(12, memBlock.Count);
            Assert.Equal(2, memBlock.Dims.Rank);
            Assert.Equal(4, memBlock.Dims[1]);
        }

        public class ColumnHintTestData
        {
            public ColumnHintTestData(TensorDimensions initialDims, int columnHint, TensorDimensions expectedDims, string comment)
            {
                InitialDims = initialDims;
                ColumnHint = columnHint;
                ExpectedDims = expectedDims;
                Comment = comment;
            }

            public TensorDimensions InitialDims { get; private set; }
            public int ColumnHint { get; private set; }
            public TensorDimensions ExpectedDims { get; private set; }
            public string Comment { get; private set; }
        }

        // ReSharper disable once MemberCanBePrivate.Global
        public static readonly TheoryData ColumnHintData = new TheoryData<ColumnHintTestData>
        {
            new ColumnHintTestData(new TensorDimensions(12), 7, new TensorDimensions(12), "ColumnHint ignored when not divisible"),
            
            new ColumnHintTestData(new TensorDimensions(12), 3, new TensorDimensions(3, 4), "ColumnHint used when divisible"),
            
            new ColumnHintTestData(new TensorDimensions(6, 2), 3, new TensorDimensions(3, 4), "CH used for matrices while count remains constant"),
        };

        [Theory, MemberData("ColumnHintData")]
        public void ColumnHintTests(ColumnHintTestData testData)
        {
            m_output.WriteLine("Running '{0}'", testData.Comment);
            
            MyAbstractMemoryBlock memBlock = GetMemBlock(testData.InitialDims);
            memBlock.ColumnHint = testData.ColumnHint;

            Assert.True(memBlock.Dims.Equals(testData.ExpectedDims));
        }

        [Fact]
        public void UseColumnHintWhenSettingCountAfterIt()
        {
            MyAbstractMemoryBlock memBlock = GetMemBlock(new TensorDimensions());

            memBlock.ColumnHint = 3;
            Assert.Equal(0, memBlock.Dims.ElementCount);
            Assert.Equal(3, memBlock.ColumnHint);
           
            memBlock.Count = 12;
            Assert.Equal(12, memBlock.Dims.ElementCount);
            Assert.Equal(2, memBlock.Dims.Rank);
            Assert.Equal(4, memBlock.Dims[1]);
        }

        public static readonly TheoryData<TensorDimensions, TensorDimensions> TranspositionData
            = new TheoryData<TensorDimensions, TensorDimensions>
        {
            { new TensorDimensions(2, 4), new TensorDimensions(4, 2) },
            { new TensorDimensions(1, 4, 3), new TensorDimensions(4, 1, 3) },
            { new TensorDimensions(4), new TensorDimensions(1, 4) },  // Transpose row vector to a column one.
            { TensorDimensions.Empty, TensorDimensions.Empty }
        };

        [Theory, MemberData("TranspositionData")]
        public void TranspositionTests(TensorDimensions initial, TensorDimensions expected)
        {
            Assert.True(initial.Transpose().Equals(expected));
        }
    }
}
