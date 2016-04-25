using System;
using System.Collections.Generic;
using VRageMath;
using World.Physics;
using Xunit;
using Rectangle = World.Physics.Rectangle;

namespace ToyWorldTests.Physics
{
    public class CollisionResolverTest
    {
        [Fact]
        public void ResolveCollissionOrthogonal()
        {
            var initPosition = new VRageMath.Vector2(0.5f,0f);
            var shape = new Rectangle(new VRageMath.Vector2(1,1));
            IForwardMovablePhysicalEntity pe = new ForwardMovablePhysicalEntity(initPosition, shape, 1, MathHelper.Pi / 2);

            //var collisionCheckerMock = new Mock<ICollisionChecker>();
            //collisionCheckerMock.Setup(x => x.CollidesWithTile(pe)).Returns(pe.Position.X < 0);

            ICollisionChecker collisionChecker = new CollisionCheckerMock();
            var movementPhysics = new MovementPhysics();
            ICollisionResolver collisionResolver = new CollisionResolver(collisionChecker, movementPhysics);

            movementPhysics.Move(pe);

            collisionResolver.ResolveCollision(pe);

            Assert.Equal(pe.Position.X, 0, 2);
            Assert.Equal(pe.Position.Y, 0, 2);
        }

        [Fact]
        public void ResolveCollission45()
        {
            var initPosition = new VRageMath.Vector2(0.5f, 0f);
            var shape = new Rectangle(new VRageMath.Vector2(1, 1));
            IForwardMovablePhysicalEntity pe = new ForwardMovablePhysicalEntity(initPosition, shape, 1, MathHelper.Pi / 4 * 3);

            //var collisionCheckerMock = new Mock<ICollisionChecker>();
            //collisionCheckerMock.Setup(x => x.CollidesWithTile(pe)).Returns(pe.Position.X < 0);

            ICollisionChecker collisionChecker = new CollisionCheckerMock();
            var movementPhysics = new MovementPhysics();
            ICollisionResolver collisionResolver = new CollisionResolver(collisionChecker, movementPhysics);

            movementPhysics.Move(pe);

            collisionResolver.ResolveCollision(pe);

            Assert.Equal(pe.Position.X, 0f, 2);
            Assert.Equal(pe.Position.Y, -0.71f, 2);
        }
    }

    internal class CollisionCheckerMock : ICollisionChecker
    {
        List<PhysicalEntity> ICollisionChecker.CollidesWithAnotherEntity(IPhysicalEntity physicalEntity)
        {
            throw new NotImplementedException();
        }

        bool ICollisionChecker.CollidesWithTile(IPhysicalEntity physicalEntity)
        {
            return physicalEntity.Position.X < 0;
        }

        float ICollisionChecker.MaximumGameObjectRadius
        {
            get { throw new NotImplementedException(); }
        }
    }
}
