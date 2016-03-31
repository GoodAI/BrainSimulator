using World.GameActions;
using World.GameActors.Tiles;
using Xunit;
using Moq;

namespace ToyWorldTests.World
{
    public class WallsTests
    {
        private readonly Wall m_wall;
        private TilesetTable m_tilesetTable;
        public WallsTests()
        {
            m_wall = new Wall(0);
            var mockTilesetTable = new Mock<TilesetTable>();
            mockTilesetTable.Setup(x => x.TileNumber(It.IsAny<string>())).Returns(0);
            mockTilesetTable.Setup(x => x.TileName(It.IsAny<int>())).Returns("");
            m_tilesetTable = mockTilesetTable.Object;
        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.009f)]
        [InlineData(0.999f)]
        [InlineData(1.0f)]
        [InlineData(3.0f)]
        public void PickaxeMakesDamageWall0(float damage)
        {
            ToUsePickaxe pickaxe = new ToUsePickaxe(m_tilesetTable) { Damage = damage };

            // Act
            var pickaxedWall = m_wall.ApplyGameAction(pickaxe, m_tilesetTable);

            if (damage >= 1)
            {
                Assert.IsType(typeof(DestroyedWall), pickaxedWall);
            }
            else if (damage > 0)
            {
                Assert.IsType(typeof(DamagedWall), pickaxedWall);
                DamagedWall damagedWall = (DamagedWall)pickaxedWall;
                Assert.True(damagedWall.Health >= 1.0f - damage);
            }
            else
            {
                Assert.IsType(typeof(Wall), pickaxedWall);
            }
            
        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.3f)]
        [InlineData(0.5f)]
        [InlineData(1.0f)]
        public void PickaxeMakesDestroyedWallFromDamagedWall(float damage)
        {
            float initialDamage = 0.5f;
            // Assert
            ToUsePickaxe pickaxe = new ToUsePickaxe(m_tilesetTable) { Damage = damage };

            DamagedWall damagedWall = new DamagedWall(initialDamage, m_tilesetTable);
            Tile pickaxedWall = damagedWall.ApplyGameAction(pickaxe, m_tilesetTable);


            if (damage + initialDamage >= 1)
            {
                Assert.IsType(typeof (DestroyedWall), pickaxedWall);
            }
            else
            {
                Assert.IsType(typeof(DamagedWall), pickaxedWall);
            }
        }
    }
}
