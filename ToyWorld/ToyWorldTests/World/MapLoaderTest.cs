using System.IO;
using System.Linq;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using VRageMath;
using World.GameActors;
using World.GameActors.GameObjects;
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

            var serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStreamReader);

            var tilesetTable = new TilesetTable(map, tilesetTableStreamReader);

            // create atlas
            m_atlas = MapLoader.LoadMap(map, tilesetTable, (GameActor actor) => { });
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

        [Fact]
        public void AvatarLoaded()
        {
            Assert.NotNull(m_atlas.Avatars.First().Value);

            IAvatar avatar = m_atlas.Avatars.First().Value;

            Assert.True(avatar.Id == 16);
            Assert.True(avatar.Name == "Pingu");
            Assert.True(avatar.PhysicalEntity.Position == new Vector2(9, 22));
            Assert.True(avatar.PhysicalEntity.Size == new Vector2(16, 16));
        }
    }
}
