using Moq;
using System;
using World.GameActors.Tiles;
using Xunit;

namespace ToyWorldTests.World
{
    public class TileTests
    {
        [Fact]
        public void NullTilesetTableThrows()
        {
            Assert.Throws<ArgumentNullException>(() => new Path(null));
        }

        [Fact]
        public void CastOnNullThrows()
        {
            Tile tile = null;
            Assert.Throws<ArgumentNullException>(() => (int)tile);
        }
    }
}
