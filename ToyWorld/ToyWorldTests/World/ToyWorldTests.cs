using Moq;
using System.Collections.Generic;
using System.IO;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
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
            var tmxStreamReader = new StreamReader(FileStreams.GetTmxMemoryStream());
            var tilesetTableStreamReader = new StreamReader(FileStreams.GetTilesetTableMemoryStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStreamReader);
            m_world = new ToyWorld(map, tilesetTableStreamReader);
        }

        [Fact]
        public void TestRegisterTicks()
        {
            AutoupdateRegister register = m_world.AutoupdateRegister;
            Mock<IAutoupdateable> mock1 = new Mock<IAutoupdateable>();
            register.Register(mock1.Object, 1);

            Assert.Equal(new List<IAutoupdateable>(), register.CurrentUpdateRequests);

            register.Tick();

            Assert.Equal(mock1.Object, register.CurrentUpdateRequests[0]);
        }

        [Fact]
        public void TestUpdateScheduled()
        {
            AutoupdateRegister register = m_world.AutoupdateRegister;

            Mock<Tile> mockTile = new Mock<Tile>();

            Mock<IAutoupdateable> mock1 = new Mock<IAutoupdateable>();
            Mock<IAutoupdateable> mock2 = new Mock<IAutoupdateable>();

            register.Register(mock1.Object, 1);
            register.Register(mock2.Object, 2);
            register.Tick();

            m_world.UpdateScheduled();

            mock1.Verify(x => x.Update(It.IsAny<Atlas>(), It.IsAny<TilesetTable>(), It.IsAny<AutoupdateRegister>()));
            mock2.Verify(x => x.Update(It.IsAny<Atlas>(), It.IsAny<TilesetTable>(), It.IsAny<AutoupdateRegister>()), Times.Never());

            m_world.UpdateScheduled();

            mock2.Verify(x => x.Update(It.IsAny<Atlas>(), It.IsAny<TilesetTable>(), It.IsAny<AutoupdateRegister>()));
        }

        [Fact]
        public void TestAvatarNames()
        {
            Assert.Contains<string>("Pingu", m_world.GetAvatarsNames());
        }
    }
}
