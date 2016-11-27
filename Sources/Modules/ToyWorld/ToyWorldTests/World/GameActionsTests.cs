using Moq;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.GameActions;
using World.GameActors;
using World.GameActors.GameObjects;
using World.GameActors.Tiles;
using Xunit;
using World.ToyWorldCore;

namespace ToyWorldTests.World
{
    public class GameActionsTests
    {
        [Fact]
        public void PickupResolveAddsToInventory()
        {
            Mock<IAtlas> atlas = new Mock<IAtlas>();
            Mock<GameActor> actor = new Mock<GameActor>();
            Mock<ICanPickGameObject> picker = actor.As<ICanPickGameObject>();
            picker.Setup(x => x.AddToInventory(It.IsAny<IPickableGameActor>()));

            Mock<GameActor> targetActor = new Mock<GameActor>();
            targetActor.As<IPickableGameActor>();

            PickUp pickUp = new PickUp(actor.Object);

            // Act
            pickUp.Resolve(new GameActorPosition(targetActor.Object, new Vector2(), LayerType.ObstacleInteractable), atlas.Object, It.IsAny<ITilesetTable>());

            // Assert
            picker.Verify(x => x.AddToInventory(It.IsAny<IPickableGameActor>()));
        }
    }
}
