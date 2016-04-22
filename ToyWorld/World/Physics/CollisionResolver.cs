using System;
using VRageMath;

namespace World.Physics
{
    public interface ICollisionResolver
    {
        void ResolveCollision(IForwardMovablePhysicalEntity physicalEntity);
    }

    public class CollisionResolver : ICollisionResolver
    {
        private readonly ICollisionChecker m_collisionChecker;
        private readonly IMovementPhysics m_movementPhysics;

        private const int BINARY_SEARCH_ITERATIONS = 16;
        private const float X_DIRECTION = (float) Math.PI / 2;
        private const float Y_DIRECTION = 0;
        private const float NEGLIGIBLE_DISTANCE = 0.001f;

        private Random random = new Random(79);

        public CollisionResolver(ICollisionChecker collisionChecker, IMovementPhysics movementPhysics)
        {
            m_collisionChecker = collisionChecker;
            m_movementPhysics = movementPhysics;
        }

        public void ResolveCollision(IForwardMovablePhysicalEntity physicalEntity)
        {
            if (!m_collisionChecker.CollidesWithTile(physicalEntity))
            {
                return;
            }
            else
            {
                // move to previous position
                physicalEntity.Position = Utils.Move(physicalEntity.Position, physicalEntity.Direction, -physicalEntity.ForwardSpeed);
            }
            FindTileFreePosition(physicalEntity);
        }

        private void FindTileFreePosition(IForwardMovablePhysicalEntity physicalEntity)
        {
            float speed = physicalEntity.ForwardSpeed;
            Vector2 previousPosition = physicalEntity.Position;
            // get back to last free position in direction counter to original direction
            TileFreePositionBinarySearch(physicalEntity, physicalEntity.ForwardSpeed, physicalEntity.Direction);

            float residueSpeed = speed - Vector2.Distance(previousPosition, physicalEntity.Position);

            if (physicalEntity.TileCollision == TileCollision.Slide)
            {
                Slide(physicalEntity, residueSpeed);
            }
            else if(physicalEntity.TileCollision == TileCollision.Bounce)
            {
                Bounce(physicalEntity, residueSpeed, 10);
            }
        }

        private void Bounce(IForwardMovablePhysicalEntity physicalEntity, float residueSpeed, int maxDepth)
        {
            Vector2 originalPosition = physicalEntity.Position;
            float originalDirection = physicalEntity.Direction;

            float maxDistance = 0;
            Vector2 bestPosition = originalPosition;
            float bestDirection = originalDirection;

            for (int i = 0; i < 2; i++)
            {
                float newDirection;
                if (i == 0)
                {
                    newDirection = MathHelper.WrapAngle(2f * MathHelper.Pi - physicalEntity.Direction);
                }
                else
                {
                    newDirection = MathHelper.WrapAngle(3f * MathHelper.Pi - physicalEntity.Direction);
                }

                TileFreePositionBinarySearch(physicalEntity, residueSpeed, newDirection);

                float distance = Vector2.Distance(originalPosition, physicalEntity.Position);
                if (maxDistance < distance)
                {
                    maxDistance = distance;
                    bestPosition = physicalEntity.Position;
                    bestDirection = newDirection;
                }

                physicalEntity.Position = originalPosition;
            }

            physicalEntity.Position = bestPosition;
            physicalEntity.Direction = bestDirection;

            if (maxDistance < residueSpeed - NEGLIGIBLE_DISTANCE && maxDepth > 0)
            {
                Bounce(physicalEntity, residueSpeed - maxDistance, maxDepth - 1);
            }
        }

        private void Slide(IForwardMovablePhysicalEntity physicalEntity, float residueSpeed)
        {
            Vector2 freePosition = physicalEntity.Position;
            float directionRads = physicalEntity.Direction;

            float xSpeed = (float)Math.Sin(directionRads) * residueSpeed;
            float ySpeed = (float)Math.Cos(directionRads) * residueSpeed;
            // position before move

            // try to move orthogonally left/right and up/down
            TileFreePositionBinarySearch(physicalEntity, xSpeed, X_DIRECTION);
            TileFreePositionBinarySearch(physicalEntity, ySpeed, Y_DIRECTION);
            Vector2 xFirstPosition = new Vector2(physicalEntity.Position.X, physicalEntity.Position.Y);

            // try to move orthogonally up/down and left/right; reset position first
            physicalEntity.Position = freePosition;
            TileFreePositionBinarySearch(physicalEntity, ySpeed, Y_DIRECTION);
            TileFreePositionBinarySearch(physicalEntity, xSpeed, X_DIRECTION);
            Vector2 yFirstPosition = new Vector2(physicalEntity.Position.X, physicalEntity.Position.Y);

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
        /// Search for position close to obstacle in given direction. Must be on stable position when starting search.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <param name="initialSpeed"></param>
        /// <param name="direction"></param>
        /// <param name="goForward">If true make step forward. Otherwise start with half step back.</param>
        private void TileFreePositionBinarySearch(IForwardMovablePhysicalEntity physicalEntity, float initialSpeed, float direction)
        {
            float speed = initialSpeed;
            bool goForward = false;

            Vector2 lastNotColliding = physicalEntity.Position;

            m_movementPhysics.Shift(physicalEntity, initialSpeed, direction);

            for (int i = 0; i < BINARY_SEARCH_ITERATIONS; i++)
            {
                if (goForward)
                {
                    m_movementPhysics.Shift(physicalEntity, speed, direction);
                }
                else
                {
                    m_movementPhysics.Shift(physicalEntity, -speed, direction);
                }
                bool colliding = m_collisionChecker.CollidesWithTile(physicalEntity);
                if (!colliding)
                {
                    lastNotColliding = physicalEntity.Position;
                }
                speed = speed / 2;
                goForward = !colliding;
            }

            physicalEntity.Position = lastNotColliding;
        }
    }
}
