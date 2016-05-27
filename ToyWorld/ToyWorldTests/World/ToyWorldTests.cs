using System;
using Moq;
using System.IO;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using World.GameActors;
using World.GameActors.Tiles;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class ToyWorldTests
    {
        private ToyWorld m_world;

        public ToyWorldTests()
        {
            Stream tmxStream = FileStreams.SmallTmx();
            StreamReader tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);
            m_world = new ToyWorld(map, tilesetTableStreamReader);
        }

        [Fact]
        public void WorldWithNullMapThrows()
        {
            Assert.Throws<ArgumentNullException>(() => new ToyWorld(null, new StreamReader(FileStreams.TilesetTableStream())));
        }

        [Fact]
        public void WorldWithNullTilesetThrows()
        {
            Stream tmxStream = FileStreams.SmallTmx();

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);

            Assert.Throws<ArgumentNullException>(() => new ToyWorld(map, null));
        }

        [Fact]
        public void WorldUpdatesAutoupdateables()
        {
            var tmxStream = FileStreams.SmallTmx();
            var tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);
            TestingToyWorld toyWorld = new TestingToyWorld(map, tilesetTableStreamReader);
            toyWorld.SetRegister(new AutoupdateRegister());

            Mock<IAutoupdateableGameActor> mock1 = new Mock<IAutoupdateableGameActor>();
            Mock<IAutoupdateableGameActor> mock2 = new Mock<IAutoupdateableGameActor>();
            toyWorld.AutoupdateRegister.Register(mock1.Object, 1);
            toyWorld.AutoupdateRegister.Register(mock2.Object, 2);

            // Act
            toyWorld.Update();

            // Assert
            mock1.Verify(x => x.Update(It.IsAny<Atlas>(), It.IsAny<TilesetTable>()));
            mock2.Verify(x => x.Update(It.IsAny<Atlas>(), It.IsAny<TilesetTable>()), Times.Never());

            // Act
            toyWorld.Update();

            // Assert
            mock2.Verify(x => x.Update(It.IsAny<Atlas>(), It.IsAny<TilesetTable>()));
        }

        [Fact]
        public void TestAvatarNames()
        {
            Assert.Contains<string>("Pingu", m_world.GetAvatarsNames());
        }

        private class TestingToyWorld : ToyWorld
        {
            public TestingToyWorld(Map tmxDeserializedMap, StreamReader tileTable) : base(tmxDeserializedMap, tileTable) { }

            public void SetRegister(AutoupdateRegister register)
            {
                AutoupdateRegister = register;
            }
        }
    }
}
