using Moq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using World.Physics;
using Xunit;

namespace ToyWorldTests.Physics
{
    public class CollisionResolverTest
    {
        public CollisionResolverTest()
        {

        }

        [Fact]
        public void ResolveCollissionOrthogonal()
        {
            var initPosition = new VRageMath.Vector2(0.5f,0f);
            var shape = new Rectangle(new VRageMath.Vector2(1,1));
            IForwardMovablePhysicalEntity pe = new ForwardMovablePhysicalEntity(initPosition, shape, 1, 180);

            //var collisionCheckerMock = new Mock<ICollisionChecker>();
            //collisionCheckerMock.Setup(x => x.CollidesWithTile(pe)).Returns(pe.Position.X < 0);

            ICollisionChecker collisionChecker = new CollisionCheckerMock();
            ICollisionResolver collisionResolver = new CollisionResolver(collisionChecker);

            pe.Move();

            collisionResolver.ResolveCollisionWithTile(pe);

            Assert.True(TestUtils.FloatEq(pe.Position.X, 0, 0.02f));
            Assert.True(TestUtils.FloatEq(pe.Position.Y, 0));
        }

        [Fact]
        public void ResolveCollission45()
        {
            var initPosition = new VRageMath.Vector2(0.5f, 0f);
            var shape = new Rectangle(new VRageMath.Vector2(1, 1));
            IForwardMovablePhysicalEntity pe = new ForwardMovablePhysicalEntity(initPosition, shape, 1, 135);

            //var collisionCheckerMock = new Mock<ICollisionChecker>();
            //collisionCheckerMock.Setup(x => x.CollidesWithTile(pe)).Returns(pe.Position.X < 0);

            ICollisionChecker collisionChecker = new CollisionCheckerMock();
            ICollisionResolver collisionResolver = new CollisionResolver(collisionChecker);

            pe.Move();

            collisionResolver.ResolveCollisionWithTile(pe);

            Assert.True(TestUtils.FloatEq(pe.Position.X, 0, 0.02f));
            Assert.True(TestUtils.FloatEq(pe.Position.Y, 0.71f, 0.02f));
        }
    }

    internal class CollisionCheckerMock : ICollisionChecker
    {

        public List<PhysicalEntity> CollidesWithAnotherEntity(IPhysicalEntity physicalEntity)
        {
            throw new NotImplementedException();
        }

        public bool CollidesWithTile(IPhysicalEntity physicalEntity)
        {
            return physicalEntity.Position.X < 0;
        }

        public float MaximumGameObjectRadius
        {
            get { throw new NotImplementedException(); }
        }
    }
}
