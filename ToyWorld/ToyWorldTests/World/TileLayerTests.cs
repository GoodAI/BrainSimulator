using System;
using VRageMath;
using World.GameActors.Tiles;
using World.GameActors.Tiles.Obstacle;
using World.GameActors.Tiles.Path;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class SimpleTileLayerTests
    {
        [Fact]
        public void InvalidSizeThrows()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SimpleTileLayer(LayerType.All, 0, 5));
            Assert.Throws<ArgumentOutOfRangeException>(() => new SimpleTileLayer(LayerType.All, 5, 0));
        }

        [Fact]
        public void TestGetActor()
        {
            SimpleTileLayer layer = new SimpleTileLayer(LayerType.All, 7, 7);
            Path path = new Path(0);
            layer.Tiles[3][2] = path;

            Assert.Equal(path, layer.GetActorAt(3, 2));
            Assert.Equal(path, layer.GetActorAt(new Vector2I(3, 2)));
        }

        [Theory]
        [InlineData(0, 0, true)]
        [InlineData(0, 1, true)]
        [InlineData(1, 0, true)]
        [InlineData(6, 6, true)]
        [InlineData(-1, 0, false)]
        [InlineData(0, -1, false)]
        [InlineData(-1, -1, false)]
        [InlineData(6, 7, false)]
        [InlineData(7, 6, false)]
        [InlineData(7, 7, false)]
        public void ActorOutOfRangeIsObstacle(int x, int y, bool valid)
        {
            SimpleTileLayer layer = new SimpleTileLayer(LayerType.All, 7, 7);

            if (valid)
                Assert.IsNotType<Obstacle>(layer.GetActorAt(x, y));
            else
                Assert.IsType<Obstacle>(layer.GetActorAt(x, y));
        }

        public void TestGetRectangle()
        {
            throw new NotImplementedException();
        }
    }
}
