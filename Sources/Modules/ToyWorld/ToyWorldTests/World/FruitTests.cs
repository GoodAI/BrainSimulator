using Moq;
using VRageMath;
using World.Atlas;
using World.GameActions;
using World.GameActors;
using World.GameActors.Tiles;
using World.GameActors.Tiles.ObstacleInteractable;
using World.ToyWorldCore;
using Xunit;

namespace ToyWorldTests.World
{
    public class FruitTests
    {
        [Fact]
        public void FruitPickUpCallDefault()
        {
            Mock<TilesetTable> mockTilesetTable = new Mock<TilesetTable>();
            Mock<IAtlas> atlas = new Mock<IAtlas>();
            Mock<GameActor> sender = new Mock<GameActor>();
            Mock<PickUp> pickUp = new Mock<PickUp>(sender.Object);
            pickUp.Setup(x => x.Resolve(It.IsAny<GameActorPosition>(), It.IsAny<IAtlas>()));

            Mock<Fruit> fruit = new Mock<Fruit>(mockTilesetTable.Object, Vector2I.Zero);

            // Act
            fruit.Object.PickUp(atlas.Object, pickUp.Object, Vector2.Zero);

            // Assert
            pickUp.Verify(x => x.Resolve(It.IsAny<GameActorPosition>(), atlas.Object));
        }
    }
}
