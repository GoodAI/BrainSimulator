using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.TypeMapping;
using Xunit;

namespace CoreTests.Memory
{
    public class MemoryBlockMetadataTests : CoreTestBase
    {
        [Fact]
        public void GetsMetadataByEnumIndexer()
        {
            var metadata = TypeMap.GetInstance<IMemoryBlockMetadata>();
            metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.BlackWhite;
            Assert.Equal(RenderingMethod.BlackWhite, metadata[MemoryBlockMetadataKeys.RenderingMethod]);
        }

        [Fact]
        public void ThrowsWithIncorrectEnumIndexer()
        {
            var metadata = TypeMap.GetInstance<IMemoryBlockMetadata>();
            Assert.Throws<KeyNotFoundException>(() => metadata[MemoryBlockMetadataKeys.RenderingMethod]);
        }

        [Fact]
        public void GetsMetadataWithTryGet()
        {
            var metadata = TypeMap.GetInstance<IMemoryBlockMetadata>();
            metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.BlackWhite;

            RenderingMethod method;
            Assert.True(metadata.TryGetValue(MemoryBlockMetadataKeys.RenderingMethod, out method));
            Assert.Equal(RenderingMethod.BlackWhite, method);
        }

        [Fact]
        public void DoesNotGetMetadataWithTryGetWithMissingKey()
        {
            var metadata = TypeMap.GetInstance<IMemoryBlockMetadata>();

            RenderingMethod method;
            Assert.False(metadata.TryGetValue(MemoryBlockMetadataKeys.RenderingMethod, out method));
        }

        [Fact]
        public void GetsMetadataWithGetDefault()
        {
            var metadata = TypeMap.GetInstance<IMemoryBlockMetadata>();
            metadata[MemoryBlockMetadataKeys.RenderingMethod] = RenderingMethod.BlackWhite;

            var method = metadata.GetOrDefault<RenderingMethod>(MemoryBlockMetadataKeys.RenderingMethod);

            Assert.Equal(RenderingMethod.BlackWhite, method);
        }

        [Fact]
        public void UsesTypeDefaultWithGetDefaultWithMissingKey()
        {
            var metadata = TypeMap.GetInstance<IMemoryBlockMetadata>();

            var method = metadata.GetOrDefault<RenderingMethod>(MemoryBlockMetadataKeys.RenderingMethod);

            Assert.Equal(default(RenderingMethod), method);
        }

        [Fact]
        public void UsesSpecifiedDefaultWithGetDefaultWithMissingKey()
        {
            var metadata = TypeMap.GetInstance<IMemoryBlockMetadata>();

            var method = metadata.GetOrDefault(MemoryBlockMetadataKeys.RenderingMethod,
                defaultValue: RenderingMethod.RGB);

            Assert.Equal(RenderingMethod.RGB, method);
        }
    }
}
