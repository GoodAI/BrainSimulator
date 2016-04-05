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
        /// <summary>
        /// Loads map from testing files
        /// </summary>
        private readonly Atlas m_atlas;
        public MapLoaderTests()
        {
            // initiate streamReaders
            var tmxMemoryStream = TestingFiles.Files.GetTmxMemoryStream();
            var tilesetTableMemoryStream = TestingFiles.Files.GetTilesetTableMemoryStream();

            var tmxStreamReader = new StreamReader(tmxMemoryStream);
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);

            var tilesetTable = new TilesetTable(tilesetTableStreamReader);

            // create atlas
            m_atlas = MapLoader.LoadMap(tmxStreamReader, tilesetTable);
        }

        [Fact]
        public void MapCanLoad()
        {
            Assert.NotNull(m_atlas);
            // at least 1 static tile expected
            Assert.True(m_atlas.StaticTilesContainer.Count > 0);
            // at least 7 tile layers
            Assert.True(m_atlas.TileLayers.Count >= 7);
        }
    }
}
