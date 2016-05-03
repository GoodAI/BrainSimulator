using System;
using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Physics;
using Xunit;

namespace ToyWorldTests.Physics
{
    public class CollisionResolverTest
    {
        [Fact]
        public void ResolveSlideCollissionOrthogonal()
        {
            var initPosition = new Vector2(0.5f,0f);
            var shape = new RectangleShape(new Vector2(1,1));
            IForwardMovablePhysicalEntity pe = new ForwardMovablePhysicalEntity(initPosition, shape, 1, MathHelper.Pi / 2);
            pe.SlideOnCollision = true;

            ICollisionChecker collisionChecker = new CollisionCheckerMock(new List<IPhysicalEntity>() {pe});
            var movementPhysics = new MovementPhysics();
            ICollisionResolver collisionResolver = new NaiveCollisionResolver(collisionChecker, movementPhysics);

            movementPhysics.Move(pe);

            collisionResolver.ResolveCollisions();

            Assert.Equal(pe.Position.X, 0, 2);
            Assert.Equal(pe.Position.Y, 0, 2);
        }

        [Fact]
        public void ResolveSlideCollission45()
        {
            var initPosition = new Vector2(0.5f, 0f);
            var shape = new RectangleShape(new Vector2(1, 1));

            float angle = 135f;

            var angleRads = MathHelper.ToRadians(angle);
            IForwardMovablePhysicalEntity pe = new ForwardMovablePhysicalEntity(initPosition, shape, 1, angleRads);
            pe.SlideOnCollision = true;

            ICollisionChecker collisionChecker = new CollisionCheckerMock(new List<IPhysicalEntity>() {pe});
            var movementPhysics = new MovementPhysics();
            ICollisionResolver collisionResolver = new NaiveCollisionResolver(collisionChecker, movementPhysics);

            movementPhysics.Move(pe);

            collisionResolver.ResolveCollisions();

            Assert.Equal(pe.Position.X, 0f, 2);
            Assert.Equal(pe.Position.Y, -0.71f, 2);
        }
    }

    internal class CollisionCheckerMock : ICollisionChecker
    {
        private readonly List<IPhysicalEntity> m_physicalEntities;

        public CollisionCheckerMock(List<IPhysicalEntity> physicalEntities)
        {
            m_physicalEntities = physicalEntities;
        }

        public bool CollidesWithTile(IPhysicalEntity physicalEntity)
        {
            return physicalEntity.Position.X < 0;
        }

        float ICollisionChecker.MaximumGameObjectRadius
        {
            get { throw new NotImplementedException(); }
        }

        public List<List<IPhysicalEntity>> CollisionGroups()
        {
            List<List<IPhysicalEntity>> list = new List<List<IPhysicalEntity>>();
            list.Add(m_physicalEntities);
            return list;
        }

        public float MaximumGameObjectSpeed
        {
            get { return 1; }
        }

        public bool Collides(List<IPhysicalEntity> physicalEntities)
        {
            return physicalEntities.Any(CollidesWithTile);
        }

        public bool Collides(IPhysicalEntity physicalEntity)
        {
            return Collides(new List<IPhysicalEntity>() {physicalEntity});
        }

        public bool CollidesWithPhysicalEntity(IPhysicalEntity physicalEntity)
        {
            throw new NotImplementedException();
        }


        public bool CollidesWithTile(IPhysicalEntity physicalEntity, float eps)
        {
            return physicalEntity.Position.X - eps < 0;
        }

        public bool CollidesWithEachOther(List<IPhysicalEntity> physicalEntities)
        {
            throw new NotImplementedException();
        }

        public List<IPhysicalEntity> CollisionThreat(IPhysicalEntity targetEntity, List<IPhysicalEntity> physicalEntities, float eps = 0)
        {
            return new List<IPhysicalEntity>();
        }
    }
}
