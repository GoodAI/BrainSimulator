using World.GameActions;
using World.GameActors.Tiles;
using World.Tiles;
using Xunit;

namespace ToyWorldTests.World
{
    public class WallsTests
    {
        private readonly Wall m_wall;
        public WallsTests()
        {
            TileSetTableParser tstp = new TileSetTableParser();
            m_wall = new Wall(tstp);
        }

        [Fact]
        public void TileTypeAssigned()
        {
            // Assert
            Assert.True(m_wall.TileType > 0);
        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.009f)]
        [InlineData(0.999f)]
        public void PickaxeMakesDamageWall0(float damage)
        {
            ToUsePickaxe pickaxe = new ToUsePickaxe { Damage = damage };

            // Act
            var pickaxedWall = m_wall.ApplyGameAction(pickaxe);

            if (damage > 0)
            {
                Assert.IsType(typeof(DamagedWall), pickaxedWall);
                DamagedWall damagedWall = (DamagedWall) pickaxedWall;
                Assert.True(damagedWall.Health >= 1.0f - damage);
            }
            else
            {
                Assert.IsType(typeof(Wall), pickaxedWall);
            }
            
        }


        [Theory]
        [InlineData(1.0f)]
        [InlineData(2.0f)]
        public void PickaxeMakesDestroyedWall(float damage)
        {
            ToUsePickaxe pickaxe = new ToUsePickaxe { Damage = damage };

            // Act
            var pickaxedWall = m_wall.ApplyGameAction(pickaxe);

            // Assert
            Assert.IsType(typeof(DestroyedWall), pickaxedWall);
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
            ToUsePickaxe pickaxe = new ToUsePickaxe { Damage = damage };

            DamagedWall damagedWall = new DamagedWall(initialDamage);
            Tile pickaxedWall = damagedWall.ApplyGameAction(pickaxe);


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
