using Moq;
using System;
using System.Collections.Generic;
using System.IO;
using TmxMapSerializer.Elements;
using TmxMapSerializer.Serializer;
using World.GameActors;
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

        [Fact]
        public void TestGetReturnsList()
        {
            TestingAutoupdateRegister register = new TestingAutoupdateRegister();

            Assert.IsType<List<IAutoupdateable>>(register.PublicCurrentUpdateRequests);
        }

        private TestingAutoupdateRegister CreateSmallRegisterWithOneObject()
        {
            TestingAutoupdateRegister register = new TestingAutoupdateRegister(2);
            Mock<IAutoupdateable> mock = new Mock<IAutoupdateable>();
            register.Register(mock.Object, 1);
            return register;
        }

        [Fact]
        public void TestRegisterAndGet()
        {
            TestingAutoupdateRegister register = new TestingAutoupdateRegister(2);
            Mock<IAutoupdateable> mock = new Mock<IAutoupdateable>();
            register.Register(mock.Object, 1);
            register.Tick();

            Assert.Equal(mock.Object, register.PublicCurrentUpdateRequests[0]);
        }

        [Fact]
        public void TestTick()
        {
            TestingAutoupdateRegister register = CreateSmallRegisterWithOneObject();
            List<IAutoupdateable> list = register.PublicCurrentUpdateRequests;
            register.Tick();

            Assert.NotEqual(list, register.PublicCurrentUpdateRequests);
        }

        [Fact]
        public void TestTickAfterEnd()
        {
            TestingAutoupdateRegister register = CreateSmallRegisterWithOneObject();
            register.Tick();
            List<IAutoupdateable> list = register.PublicCurrentUpdateRequests;
            register.Tick();
            register.Tick();

            Assert.Equal(list, register.PublicCurrentUpdateRequests);
        }

        [Fact]
        public void TestSchedulAfterEnd()
        {
            TestingAutoupdateRegister register = new TestingAutoupdateRegister(2);
            Mock<IAutoupdateable> mock = new Mock<IAutoupdateable>();
            register.Register(mock.Object, 2);
            register.Tick();
            register.Tick();

            Assert.Equal(mock.Object, register.PublicCurrentUpdateRequests[0]);
        }

        [Fact]
        public void TestUpdateItems()
        {
            Mock<IAtlas> mockAtlas = new Mock<IAtlas>();

            AutoupdateRegister register = new AutoupdateRegister();
            Mock<IAutoupdateable> mock1 = new Mock<IAutoupdateable>();
            mock1.Setup(x => x.Update(It.IsAny<IAtlas>(), It.IsAny<TilesetTable>()));
            Mock<IAutoupdateable> mock2 = new Mock<IAutoupdateable>();
            mock2.Setup(x => x.Update(It.IsAny<IAtlas>(), It.IsAny<TilesetTable>()));
            register.Register(mock1.Object, 1);
            register.Register(mock2.Object, 2);

            // Act
            register.Tick();
            register.UpdateItems(mockAtlas.Object, It.IsAny<TilesetTable>());

            // Assert
            mock1.Verify(x => x.Update(It.IsAny<IAtlas>(), It.IsAny<TilesetTable>()));
            mock2.Verify(x => x.Update(It.IsAny<IAtlas>(), It.IsAny<TilesetTable>()), Times.Never());

            // Act
            register.Tick();
            register.UpdateItems(mockAtlas.Object, It.IsAny<TilesetTable>());

            // Assert
            mock2.Verify(x => x.Update(It.IsAny<IAtlas>(), It.IsAny<TilesetTable>()));
        }

        private class TestingAutoupdateRegister : AutoupdateRegister
        {
            public List<IAutoupdateable> PublicCurrentUpdateRequests
            {
                get { return CurrentUpdateRequests; }
            }

            public TestingAutoupdateRegister(int registerSize = 100) : base(registerSize) { }
        }
    }
}
