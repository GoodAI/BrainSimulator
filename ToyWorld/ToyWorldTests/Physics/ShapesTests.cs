using System.Collections.Generic;
using World.Physics;
using Xunit;

namespace ToyWorldTests.Physics
{
    public class ShapesTests
    {
        [Theory]
        [InlineData(0.2, 0.2, 0.5)]
        [InlineData(0, 0, 1.5)]
        [InlineData(0, 0, 2)]
        [InlineData(0.1, 0.1, 1.5)]
        public void RectangleCoverTiles(float x, float y, float size)
        {
            var rectangle = new Rectangle(new VRageMath.Vector2(size, size));
            List<VRageMath.Vector2I> coverTiles = rectangle.CoverTiles(new VRageMath.Vector2(x, y));

            if (TestUtils.FloatEq(x, 0.2f) && TestUtils.FloatEq(y, 0.2f) && TestUtils.FloatEq(size, 0.5f))
            {
                Assert.True(coverTiles.Count == 1);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
            }

            if (TestUtils.FloatEq(x, 0f) && TestUtils.FloatEq(y, 0f) && TestUtils.FloatEq(size, 1.5f))
            {
                Assert.True(coverTiles.Count == 4);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }

            if (TestUtils.FloatEq(x, 0f) && TestUtils.FloatEq(y, 0f) && TestUtils.FloatEq(size, 2f))
            {
                Assert.True(coverTiles.Count == 4);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }

            if (TestUtils.FloatEq(x, 0.1f) && TestUtils.FloatEq(y, 0.1f) && TestUtils.FloatEq(size, 1.5f))
            {
                Assert.True(coverTiles.Count == 4);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }
        }

        [Theory]
        [InlineData(0.7, 0.7, 0.2)]
        [InlineData(1.2, 1.2, 0.7)]
        [InlineData(1, 1, 0.8)]
        [InlineData(1, 1, 1)]
        [InlineData(1.1, 1.1, 0.8)]
        [InlineData(2, 2, 1.2)]
        [InlineData(1.5, 2, 1.05)]
        public void CircleCoverTiles(float x, float y, float radius)
        {
            var rectangle = new Circle(radius);
            List<VRageMath.Vector2I> coverTiles = rectangle.CoverTiles(new VRageMath.Vector2(x - radius, y - radius));

            if (TestUtils.FloatEq(x, 0.7f) && TestUtils.FloatEq(y, 0.7f) && TestUtils.FloatEq(radius, 0.1f))
            {
                Assert.True(coverTiles.Count == 1);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
            }

            if (TestUtils.FloatEq(x, 1.2f) && TestUtils.FloatEq(y, 1.2f) && TestUtils.FloatEq(radius, 0.7f))
            {
                Assert.True(coverTiles.Count == 4);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }

            if (TestUtils.FloatEq(x, 1f) && TestUtils.FloatEq(y, 1f) && TestUtils.FloatEq(y, 0.8f))
            {
                Assert.True(coverTiles.Count == 4);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }

            if (TestUtils.FloatEq(x, 1f) && TestUtils.FloatEq(y, 1f) && TestUtils.FloatEq(radius, 1f))
            {
                Assert.True(coverTiles.Count == 4);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }

            if (TestUtils.FloatEq(x, 1.1f) && TestUtils.FloatEq(y, 1.1f) && TestUtils.FloatEq(radius, 0.8f))
            {
                Assert.True(coverTiles.Count == 4);
                Assert.True(coverTiles.Exists(a => a == VRageMath.Vector2.Zero));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }

            if (TestUtils.FloatEq(x, 2f) && TestUtils.FloatEq(y, 2f) && TestUtils.FloatEq(radius, 1.2f))
            {
                Assert.True(coverTiles.Count == 12);
                Assert.True(!coverTiles.Exists(a => a == new VRageMath.Vector2(0, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }

            if (TestUtils.FloatEq(x, 1.5f) && TestUtils.FloatEq(y, 2f) && TestUtils.FloatEq(radius, 1.05f))
            {
                Assert.True(coverTiles.Count == 8);
                Assert.True(!coverTiles.Exists(a => a == new VRageMath.Vector2(0, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(0, 1)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 0)));
                Assert.True(coverTiles.Exists(a => a == new VRageMath.Vector2(1, 1)));
                return;
            }
        }
    }
}
