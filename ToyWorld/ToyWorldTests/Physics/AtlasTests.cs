using GoodAI.ToyWorld.Control;
using Moq;
using Physics;
using World.GameActors.GameObjects;
using Xunit;

namespace ToyWorldTests.Physics
{
    class MovementPhysicsTest
    {
        public MovementPhysicsTest()
        {

        }

        [Fact]
        public void TestMoveForward()
        {
            var movableMock = new Mock<GameObject>();
            movableMock.As<IMovable>();
            var movable = movableMock.Object;
            movable.ForwardSpeed = 1f;
            movable.RotationSpeed = 1f;
            MovementPhysics.Move(movable);
            movable.ForwardSpeed > 1;

        }
    }
}
