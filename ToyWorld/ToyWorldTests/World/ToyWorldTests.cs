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
            var tmxStream = FileStreams.SmallTmx();
            var tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);
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
        public void TestAvatarNames()
        {
            Assert.Contains<string>("Pingu", m_world.GetAvatarsNames());
        }
    }
}
