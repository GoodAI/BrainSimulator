using World.Tiles;
using Xunit;

namespace ToyWorldTests.World
{
    public class TileTests
    {
        [Fact]
        public void PickaxeDamageWall()
        {
            // Arrange
            Wall wall = new Wall();
            GameAction pickaxe = new UsePickaxe();

            // Act
            var pickaxedWall = wall.TransformTo(pickaxe);

            // Assert
            Assert.IsType(typeof(DamagedWall), pickaxedWall);
        }
    }
}
