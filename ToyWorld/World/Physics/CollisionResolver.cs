using System;
using System.Diagnostics;
using VRageMath;

namespace World.Physics
{
    public interface ICollisionResolver
    {
        void ResolveCollision(IForwardMovablePhysicalEntity physicalEntity);
    }

    public class CollisionResolver : ICollisionResolver
    {
        ICollisionChecker m_collisionChecker;

        private const int BINARY_SEARCH_ITERATIONS = 16;
        private const float X_DIRECTION = 0;
        private const float Y_DIRECTION = 90;

        public CollisionResolver(ICollisionChecker collisionChecker)
        {
            m_collisionChecker = collisionChecker;
        }

        public void ResolveCollision(IForwardMovablePhysicalEntity physicalEntity)
        {
            var directionRads = MathHelper.ToRadians(physicalEntity.Direction);
            var speed = physicalEntity.ForwardSpeed;
            var collidingPosition = physicalEntity.Position;
            
            FindTileFreePosition(physicalEntity);
        }

        private void FindTileFreePosition(IForwardMovablePhysicalEntity physicalEntity)
        {
            var speed = physicalEntity.ForwardSpeed;
            var previousPosition = Utils.Move(physicalEntity.Position, physicalEntity.Direction, - speed);
            // get back to last free position in direction counter to original direction
            TileFreePositionBinarySearch(physicalEntity, physicalEntity.ForwardSpeed, physicalEntity.Direction, false);

            var freePosition = physicalEntity.Position;
            var directionRads = MathHelper.ToRadians(physicalEntity.Direction);
            var residueSpeed = speed - Vector2.Distance(previousPosition, freePosition);
            float xSpeed = (float)Math.Cos(directionRads) * residueSpeed;
            float ySpeed = (float)Math.Sin(directionRads) * residueSpeed;
            // position before move
            
            // try to move orthogonally left/right and up/down
            TileFreePositionBinarySearch(physicalEntity, xSpeed, X_DIRECTION, true);
            TileFreePositionBinarySearch(physicalEntity, ySpeed, Y_DIRECTION, true);
            var xFirstPosition = new Vector2(physicalEntity.Position.X, physicalEntity.Position.Y);

            // try to move orthogonally up/down and left/right; reset position first
            physicalEntity.Position = freePosition;
            TileFreePositionBinarySearch(physicalEntity, ySpeed, Y_DIRECTION, true);
            TileFreePositionBinarySearch(physicalEntity, xSpeed, X_DIRECTION, true);
            var yFirstPosition = new Vector2(physicalEntity.Position.X, physicalEntity.Position.Y);

            // farther position is chosen
            if (Vector2.Distance(freePosition, xFirstPosition) > Vector2.Distance(freePosition, yFirstPosition))
            {
                physicalEntity.Position = xFirstPosition;
            }
            else
            {
                physicalEntity.Position = yFirstPosition;
            }
        }

        /// <summary>
        /// Search for position close to obstacle in given direction.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <param name="initialSpeed"></param>
        /// <param name="direction"></param>
        /// <param name="goForward">If true make step forward. Otherwise start with half step back.</param>
        private void TileFreePositionBinarySearch(IForwardMovablePhysicalEntity physicalEntity, float initialSpeed, float direction, bool goForward)
        {
            var speed = initialSpeed;

            if (goForward)
            {
                physicalEntity.Move(initialSpeed, direction);
                goForward = false;
            }

            var colliding = m_collisionChecker.CollidesWithTile(physicalEntity);
            if (!colliding)
            {
                return;
            }

            Vector2 lastNotColliding = Vector2.PositiveInfinity;
            for (int i = 0; i < BINARY_SEARCH_ITERATIONS; i++)
            {
                if (goForward)
                {
                    physicalEntity.Move(speed, direction);
                }
                else
                {
                    physicalEntity.Move(-speed, direction);
                }
                colliding = m_collisionChecker.CollidesWithTile(physicalEntity);
                if (!colliding)
                {
                    lastNotColliding = physicalEntity.Position;
                }
                speed = speed / 2;
                goForward = !colliding;
            }
            Debug.Assert(lastNotColliding != Vector2.PositiveInfinity);
            physicalEntity.Position = lastNotColliding;
        }
    }
}
