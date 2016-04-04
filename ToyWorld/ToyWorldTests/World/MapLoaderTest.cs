using System.IO;
using System.Security.Cryptography.X509Certificates;
using GoodAI.ToyWorld.Control;
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
            var tmxMemoryStream = TestingFiles.Files.GetTmxMemoryStream();
            var tilesetTableMemoryStream = TestingFiles.Files.GetTilesetTableMemoryStream();

            var tmxStreamReader = new StreamReader(tmxMemoryStream);
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);

            var tilesetTable = new TilesetTable(tilesetTableStreamReader);

            m_atlas = MapLoader.LoadMap(tmxStreamReader, tilesetTable);
        }

        [Fact]
        public void MapCanLoad()
        {
            Assert.True(true);
        }
    }
}
