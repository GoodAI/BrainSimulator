using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GoodAI.Core.Memory;
using GoodAI.Core.Observers;
using Xunit;

namespace CoreTests.Observers
{
    public class TextureSizeTests
    {
        public class CustomDimsTestCase
        {
            public CustomDimsTestCase(TensorDimensions dims, CustomDimensionsHint customDims,
                RenderingMethod method, int vectorElements, Size expectedSize)
            {
                Dims = dims;
                CustomDims = customDims;
                Method = method;
                VectorElements = vectorElements;
                ExpectedSize = expectedSize;
            }

            public TensorDimensions Dims { get; private set; }
            public CustomDimensionsHint CustomDims { get; private set; }
            public RenderingMethod Method { get; private set; }
            public int VectorElements { get; private set; }
            public Size ExpectedSize { get; private set; }
        }

        // ReSharper disable once MemberCanBePrivate.Global
        public static readonly TheoryData CustomDimsData = new TheoryData<CustomDimsTestCase>
        {
            // Respects 2D dimensions.
            new CustomDimsTestCase(new TensorDimensions(5, 70), CustomDimensionsHint.Empty,
                RenderingMethod.RedGreenScale, 1, new Size(5, 70)),

            // Square-roots column vector.
            new CustomDimsTestCase(new TensorDimensions(1, 25), CustomDimensionsHint.Empty,
                RenderingMethod.RedGreenScale, 1, new Size(5, 5)),

            // Square-roots row vector.
            new CustomDimsTestCase(new TensorDimensions(25), CustomDimensionsHint.Empty,
                RenderingMethod.RedGreenScale, 1, new Size(5, 5)),

            // Keeps short column vector.
            new CustomDimsTestCase(new TensorDimensions(1, 10), CustomDimensionsHint.Empty,
                RenderingMethod.RedGreenScale, 1, new Size(1, 10)),

            // Keeps short row vector.
            new CustomDimsTestCase(new TensorDimensions(7), CustomDimensionsHint.Empty,
                RenderingMethod.RedGreenScale, 1, new Size(7, 1)),

            // Applies custom dimensions.
            new CustomDimsTestCase(new TensorDimensions(2, 3, 5, 7), new CustomDimensionsHint(6, 35), 
                RenderingMethod.RedGreenScale, 1, new Size(6, 35)),

            // Square-roots a vector when the hint cannot be applied.
            new CustomDimsTestCase(new TensorDimensions(25, 1), CustomDimensionsHint.Parse("7, 1, *"),
                RenderingMethod.RedGreenScale, 1, new Size(5, 5)),

            // Collapses the last dimension in RGB rendering method.
            new CustomDimsTestCase(new TensorDimensions(100, 100, 1, 3), CustomDimensionsHint.Empty,
                RenderingMethod.RGB, 1, new Size(100, 100))

            // TODO: Add more cases with Vector and RGB rendering methods.
        };

        [Theory, MemberData("CustomDimsData")]
        public void CustomTextureSizeTests(CustomDimsTestCase data)
        {
            string warning;

            Size actualSize = MyMemoryBlockObserver.ComputeCustomTextureSize(data.Dims, data.CustomDims,
                data.Method, data.VectorElements, out warning);

            Assert.Equal(data.ExpectedSize, actualSize);
        }
    }
}
