using System;
using System.IO;
using World.GameActions;
using World.GameActors.Tiles;
using Xunit;
using Moq;
using VRageMath;
using World.ToyWorldCore;
using World.GameActors;

namespace ToyWorldTests.World
{
    public class WallsTests
    {
        private readonly Wall m_wall;
        private readonly TilesetTable m_tilesetTable;
        public WallsTests()
        {
            Mock<TilesetTable> mockTilesetTable = new Mock<TilesetTable>();
            mockTilesetTable.Setup(x => x.TileNumber(It.IsAny<string>())).Returns(0);
            mockTilesetTable.Setup(x => x.TileName(It.IsAny<int>())).Returns("");
            m_tilesetTable = mockTilesetTable.Object;

            m_wall = new Wall(m_tilesetTable);
        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.009f)]
        [InlineData(0.999f)]
        [InlineData(1.0f)]
        [InlineData(3.0f)]
        public void PickaxeMakesDamageWall0(float damage)
        {
            // Arrange
            Mock<GameActor> actorMock = new Mock<GameActor>();
            ToUsePickaxe pickaxe = new ToUsePickaxe(actorMock.Object) { Damage = damage };
            Mock<IAtlas> atlasMock = new Mock<IAtlas>();
            GameActor pickaxedWall = null;
            atlasMock.Setup(x => x.ReplaceWith(It.IsAny<GameActorPosition>(), It.IsAny<GameActor>())).
                Callback((GameActorPosition original, GameActor replacement) => pickaxedWall = replacement);

            // Act
            m_wall.ApplyGameAction(atlasMock.Object, pickaxe, new Vector2I(), m_tilesetTable);

            // Assert
            if (damage >= 1)
                Assert.IsType(typeof(DestroyedWall), pickaxedWall);
            else if (damage > 0)
            {
                Assert.IsType(typeof(DamagedWall), pickaxedWall);
                DamagedWall damagedWall = (DamagedWall)pickaxedWall;
                Assert.True(damagedWall.Health >= 1.0f - damage);
            }
            else
                Assert.Equal(null, pickaxedWall);

        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.3f)]
        [InlineData(0.5f)]
        [InlineData(1.0f)]
        public void PickaxeMakesDestroyedWallFromDamagedWall(float damage)
        {
            // Arrange
            Mock<GameActor> actorMock = new Mock<GameActor>();
            Mock<IAtlas> atlasMock = new Mock<IAtlas>();
            GameActor pickaxedWall = null;
            atlasMock.Setup(x => x.ReplaceWith(It.IsAny<GameActorPosition>(), It.IsAny<GameActor>())).
                Callback((GameActorPosition original, GameActor replacement) => pickaxedWall = replacement);

            ToUsePickaxe pickaxe = new ToUsePickaxe(actorMock.Object) { Damage = damage };
            float initialDamage = 0.5f;
            DamagedWall damagedWall = new DamagedWall(initialDamage, m_tilesetTable);

            // Act
            damagedWall.ApplyGameAction(atlasMock.Object, pickaxe, new Vector2I(), m_tilesetTable);

            // Assert
            if (damage + initialDamage >= 1)
                Assert.IsType(typeof(DestroyedWall), pickaxedWall);
            else
                Assert.Equal(null, pickaxedWall);
        }

        [Fact]
        public void NonPickaxeActionDoesNothing()
        {
            Mock<GameActor> sender = new Mock<GameActor>();
            Mock<GameAction> action = new Mock<GameAction>(sender.Object);
            Mock<IAtlas> atlas = new Mock<IAtlas>();
            atlas.Setup(x => x.ReplaceWith(It.IsAny<GameActorPosition>(), It.IsAny<GameActor>()));

            // Act
            m_wall.ApplyGameAction(atlas.Object, action.Object, new Vector2I(), null);

            // Assert
            atlas.Verify(x => x.ReplaceWith(It.IsAny<GameActorPosition>(), It.IsAny<GameActor>()), Times.Never());
        }
    }
}
