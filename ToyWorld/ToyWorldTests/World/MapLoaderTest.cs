using System.IO;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
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
            var tmxMemoryStream = FileStreams.GetTmxMemoryStream();
            var tilesetTableMemoryStream = FileStreams.GetTilesetTableMemoryStream();

            var tmxStreamReader = new StreamReader(tmxMemoryStream);
            var tilesetTableStreamReader = new StreamReader(tilesetTableMemoryStream);

            var tilesetTable = new TilesetTable(tilesetTableStreamReader);

            var serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStreamReader);

            // create atlas
            m_atlas = MapLoader.LoadMap(map, tilesetTable);
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
