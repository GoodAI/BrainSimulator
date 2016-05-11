using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using World.GameActors.Tiles.Background;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class AtlasTests
    {
        private readonly Atlas m_atlas;

        public AtlasTests()
        {
            Stream tmxStream = FileStreams.SmallPickupTmx();
            StreamReader tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);
            ToyWorld world = new ToyWorld(map, tilesetTableStreamReader);

            m_atlas = world.Atlas;
        }

        [Fact]
        public void NullAvatarThrows()
        {
            Assert.Throws<ArgumentNullException>(() => m_atlas.AddAvatar(null));
        }

        [Fact]
        public void TestActorsAt()
        {
            List<GameActorPosition> results = m_atlas.ActorsAt(2, 2).ToList();

            Assert.IsType<Background>(results[0].Actor);
            Assert.IsType<Avatar>(results[1].Actor);
        }

        [Fact]
        public void TestInteractableActorsAt()
        {
            List<GameActorPosition> results = m_atlas.ActorsAt(2, 0, LayerType.Interactable).ToList();

            Assert.IsType<Apple>(results[0].Actor);
        }

        [Fact]
        public void TestActorsInFrontOf()
        {
            IAvatar avatar = m_atlas.GetAvatars()[0];

            List<GameActorPosition> results = m_atlas.ActorsInFrontOf(avatar).ToList();

            Assert.IsType<Background>(results[0].Actor);
            Assert.IsType<Apple>(results[1].Actor);
        }
    }
}
