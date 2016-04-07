using Moq;
using VRageMath;
using World.Physics;
using Xunit;

namespace ToyWorldTests.Physics
{
    public class MovementPhysicsTest
    {
        [Theory]
        [InlineData(0, 0)]
        [InlineData(0, 90)]
        [InlineData(0, 180)]
        [InlineData(1, 0)]
        [InlineData(1, 90)]
        [InlineData(1, 180)]
        [InlineData(1, 270)]
        [InlineData(1, 45)]
        [InlineData(-1, 45)]
        public void TestMoveForward(float speed, float direction)
        {
            var startingPosition = new Vector2(5, 5);

            var movableMock = new Mock<IMovableDirectablePhysicalEntity>();
            /*movableMock.Setup(x => x.Position).Returns(startingPosition);
            movableMock.Setup(x => x.ForwardSpeed).Returns(speed); 
            movableMock.Setup(x => x.Direction).Returns(direction);*/

            movableMock.SetupAllProperties();
            movableMock.Object.Position = startingPosition;
            movableMock.Object.ForwardSpeed = speed;
            movableMock.Object.Direction = direction;


            IMovableDirectablePhysicalEntity movable = movableMock.Object;

            MovementPhysics.Move(movable);


            if (speed == 0f)
            {
                Assert.True(movable.Position == startingPosition);
            }
            else if (speed == 1f)
            {
                switch ((int) direction)
                {
                    case 0:
                        Assert.True(CompareVectors(movable.Position, new Vector2(6, 5)));
                        break;
                    case 90:
                        Assert.True(CompareVectors(movable.Position, new Vector2(5, 6)));
                        break;
                    case 180:
                        Assert.True(CompareVectors(movable.Position, new Vector2(4, 5)));
                        break;
                    case 270:
                        Assert.True(CompareVectors(movable.Position, new Vector2(5, 4)));
                        break;
                    case 45:
                        Assert.True(movable.Position.X > 5.5f && movable.Position.Y > 5.5f);
                        break;
                }
            }
            else if (speed == -1f)
            {
                if (direction == 45)
                {
                    Assert.True(movable.Position.X < 4.5f && movable.Position.Y < 4.5f);
                }
            }
        }

        private static bool CompareVectors(Vector2 v1, Vector2 v2)
        {
            var maxError = Vector2.One/100000;
            return (v2 - v1).Length() < maxError.Length();
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(-1)]
        public void TestRotate(float rotationSpeed)
        {
            var startingDirection = 90;

            var movableMock = new Mock<IMovableDirectablePhysicalEntity>();
            /*movableMock.Setup(x => x.Position).Returns(startingPosition);
            movableMock.Setup(x => x.ForwardSpeed).Returns(speed); 
            movableMock.Setup(x => x.Direction).Returns(direction);*/

            movableMock.SetupAllProperties();
            movableMock.Object.Direction = startingDirection;
            movableMock.Object.RotationSpeed = rotationSpeed;


            IMovableDirectablePhysicalEntity movable = movableMock.Object;

            MovementPhysics.Move(movable);

            if (rotationSpeed == 0f)
            {
                Assert.True(movable.Direction == startingDirection);
            }
            else if (rotationSpeed == 1f)
            {
                Assert.True(movable.Direction == 91);
            }
            else if (rotationSpeed == -1f)
            {
                Assert.True(movable.Direction == 89);
            }
        }
    }
}
