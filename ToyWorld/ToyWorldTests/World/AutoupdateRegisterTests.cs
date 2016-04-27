using Moq;
using System;
using System.Collections.Generic;
using System.IO;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using World.GameActors.Tiles;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class AutoupdateRegisterTests
    {
        [Fact]
        public void TestCreateWithSize()
        {
            AutoupdateRegister register = new AutoupdateRegister(7);

            Assert.Equal(7, register.Size);
        }

        [Fact]
        public void TestCreateWithDefaultSize()
        {
            AutoupdateRegister register = new AutoupdateRegister();

            Assert.Equal(true, register.Size > 0);
        }

        [Fact]
        public void TestCreateWithZeroSizeThrows()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new AutoupdateRegister(-2));
            Assert.Throws<ArgumentOutOfRangeException>(() => new AutoupdateRegister(0));
        }

        [Fact]
        public void TestCreateWithNegativeSizeThrows()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new AutoupdateRegister(-2));
        }

        [Fact]
        public void TestRegisterNullThrows()
        {
            AutoupdateRegister register = new AutoupdateRegister();

            Assert.Throws<ArgumentNullException>(() => register.Register(null));
        }

        [Fact]
        public void TestRegisterForThisStepThrows()
        {
            AutoupdateRegister register = new AutoupdateRegister();
            Mock<IAutoupdateable> mock = new Mock<IAutoupdateable>();

            Assert.Throws<ArgumentOutOfRangeException>(() => register.Register(mock.Object, 0));
        }

        public void TestGetReturnsList()
        {
            AutoupdateRegister register = new AutoupdateRegister();

            Assert.IsType<List<IAutoupdateable>>(register.CurrentUpdateRequests);
        }

        private AutoupdateRegister CreateSmallRegisterWithOneObject()
        {
            AutoupdateRegister register = new AutoupdateRegister(2);
            Mock<IAutoupdateable> mock = new Mock<IAutoupdateable>();
            register.Register(mock.Object, 1);
            return register;
        }

        [Fact]
        public void TestRegisterAndGet()
        {
            AutoupdateRegister register = new AutoupdateRegister(2);
            Mock<IAutoupdateable> mock = new Mock<IAutoupdateable>();
            register.Register(mock.Object, 1);
            register.Tick();

            Assert.Equal(mock.Object, register.CurrentUpdateRequests[0]);
        }

        [Fact]
        public void TestTick()
        {
            AutoupdateRegister register = CreateSmallRegisterWithOneObject();
            List<IAutoupdateable> list = register.CurrentUpdateRequests;
            register.Tick();

            Assert.NotEqual(list, register.CurrentUpdateRequests);
        }

        [Fact]
        public void TestTickAfterEnd()
        {
            AutoupdateRegister register = CreateSmallRegisterWithOneObject();
            register.Tick();
            List<IAutoupdateable> list = register.CurrentUpdateRequests;
            register.Tick();
            register.Tick();

            Assert.Equal(list, register.CurrentUpdateRequests);
        }

        [Fact]
        public void TestSchedulAfterEnd()
        {
            AutoupdateRegister register = new AutoupdateRegister(2);
            Mock<IAutoupdateable> mock = new Mock<IAutoupdateable>();
            register.Register(mock.Object, 2);
            register.Tick();
            register.Tick();

            Assert.Equal(mock.Object, register.CurrentUpdateRequests[0]);
        }

        [Fact]
        public void TestUpdateItems()
        {
            var tmxStream = FileStreams.SmallTmx();
            var tilesetTableStreamReader = new StreamReader(FileStreams.TilesetTableStream());

            TmxSerializer serializer = new TmxSerializer();
            Map map = serializer.Deserialize(tmxStream);
            ToyWorld world = new ToyWorld(map, tilesetTableStreamReader);

            AutoupdateRegister register = new AutoupdateRegister();
            Mock<IAutoupdateable> mock1 = new Mock<IAutoupdateable>();
            Mock<IAutoupdateable> mock2 = new Mock<IAutoupdateable>();
            register.Register(mock1.Object, 1);
            register.Register(mock2.Object, 2);
            register.Tick();
            register.UpdateItems(world.Atlas);

            mock1.Verify(x => x.Update(It.IsAny<Atlas>()));
            mock2.Verify(x => x.Update(It.IsAny<Atlas>()), Times.Never());

            register.Tick();
            register.UpdateItems(world.Atlas);

            mock2.Verify(x => x.Update(It.IsAny<Atlas>()));
        }
    }
}
