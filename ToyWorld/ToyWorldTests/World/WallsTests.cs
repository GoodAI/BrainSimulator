using World.GameActions;
using World.Tiles;
using Xunit;

namespace ToyWorldTests.World
{
    public class WallsTests
    {
        [Fact]
        public void CreatingTile()
        {
            // Arrange
            Wall wall = new Wall();

            // Assert
            Assert.True(wall.TileType > 0);
        }

        [Fact]
        public void PickaxeMakesDamageWall0()
        {
            // Arrange
            Wall wall = new Wall();
            ToUsePickaxe pickaxe = new ToUsePickaxe {Damage = 0.009f};

            // Act
            var pickaxedWall = wall.ApplyGameAction(pickaxe);

            // Assert
            Assert.IsType(typeof (DamagedWall), pickaxedWall);

            DamagedWall damagedWall = (DamagedWall) pickaxedWall;
            Assert.True(damagedWall.Health >= 0.99f);
        }

        [Fact]
        public void PickaxeMakesDamageWall1()
        {
            // Arrange
            Wall wall = new Wall();
            ToUsePickaxe pickaxe = new ToUsePickaxe {Damage = 0.999f};

            // Act
            var pickaxedWall = wall.ApplyGameAction(pickaxe);

            // Assert
            Assert.IsType(typeof(DamagedWall), pickaxedWall);

            DamagedWall damagedWall = (DamagedWall)pickaxedWall;
            Assert.True(damagedWall.Health <= 0.01f);
        }


        [Fact]
        public void PickaxeMakesDestroyedWall0()
        {
            // Arrange
            Wall wall = new Wall();
            ToUsePickaxe pickaxe = new ToUsePickaxe {Damage = 1.0f};

            // Act
            var pickaxedWall = wall.ApplyGameAction(pickaxe);

            // Assert
            Assert.IsType(typeof(DestroyedWall), pickaxedWall);
        }

        [Fact]
        public void PickaxeMakesDestroyedWall1()
        {
            // Arrange
            Wall wall = new Wall();
            ToUsePickaxe pickaxe = new ToUsePickaxe {Damage = 1.8f};

            // Act
            var pickaxedWall = wall.ApplyGameAction(pickaxe);

            // Assert
            Assert.IsType(typeof(DestroyedWall), pickaxedWall);
        }
    }
}
