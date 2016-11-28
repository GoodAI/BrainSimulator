using System;
using World.GameActors.Tiles;
using World.GameActors.Tiles.Background;
using Xunit;

namespace ToyWorldTests.World
{
    public class TileTests
    {
        [Fact]
        public void NullTilesetTableThrows()
        {
            Assert.Throws<ArgumentNullException>(() => new PathTile(null));
        }

        [Fact]
        public void CastOnNullThrows()
        {
            Tile tile = null;
            Assert.Throws<ArgumentNullException>(() => (int)tile);
        }
    }
}
