using World.GameActors.Tiles;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class MapLoaderTests
    {
        private Atlas m_atlas;
        public MapLoaderTests()
        {
            MapLoader mapLoader = new MapLoader();
            m_atlas = mapLoader.LoadMap(@"..\..\..\TestFiles\mockup999_pantry_world.tmx",
                new TilesetTable(@"GameActors\Tiles\Tilesets\TilesetTable.csv"));
        }

        [Fact]
        public void MapCanLoad()
        {
            Assert.True(true);
        }
    }
}
