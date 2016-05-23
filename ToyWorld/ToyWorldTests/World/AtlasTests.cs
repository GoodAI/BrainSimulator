using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using VRageMath;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles.Background;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class AtlasTests
    {
        private readonly IAtlas m_atlas;

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
            List<GameActorPosition> results = m_atlas.ActorsAt(new Vector2(2,2)).ToList();

            Assert.IsType<Background>(results[0].Actor);
            Assert.IsType<Avatar>(results[1].Actor);
        }

        [Fact]
        public void TestInteractableActorsAt()
        {
            List<GameActorPosition> results = m_atlas.ActorsAt(new Vector2(2,0), LayerType.Interactables).ToList();

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

        [Theory]
        [InlineData(0, 0)]
        [InlineData(2, 0.4)]
        [InlineData(5, 1)]
        [InlineData(7, 0.6)]
        [InlineData(10, 0)]
        [InlineData(2, 0.4)]
        public void TestDayCycle(int seconds, float result)
        {
            m_atlas.DayLength = new TimeSpan(0,0,0,10);

            m_atlas.IncrementTime(seconds: seconds);

            float light = m_atlas.Light;
            Assert.Equal(light, result, 1);
        }

        [Theory]
        [InlineData(0, 0)]
        [InlineData(2, 0.4)]
        [InlineData(5, 1)]
        [InlineData(7, 0.6)]
        [InlineData(10, 0)]
        [InlineData(2, 0.4)]
        public void TestYearCycle(int seconds, float result)
        {
            m_atlas.YearLength = new TimeSpan(0, 0, 0, 10);

            m_atlas.IncrementTime(seconds: seconds);

            float summer = m_atlas.Summer;
            Assert.Equal(summer, result, 1);
        }
    }
}
