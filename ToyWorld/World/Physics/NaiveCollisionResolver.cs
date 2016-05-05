using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using VRageMath;

namespace World.Physics
{
    public class NaiveCollisionResolver : ICollisionResolver
    {
        private readonly ICollisionChecker m_collisionChecker;
        private readonly IMovementPhysics m_movementPhysics;

        private const int BINARY_SEARCH_ITERATIONS = 14;
        private const float X_DIRECTION = (float) Math.PI / 2;
        private const float Y_DIRECTION = 0;
        private const float NEGLIGIBLE_TIME = 0.001f;
        private const float NEGLIGIBLE_DISTANCE = 0.001f;
        private const int MAXIMUM_RESOLVE_TRIES = 10;

        public NaiveCollisionResolver(ICollisionChecker collisionChecker, IMovementPhysics movementPhysics)
        {
            m_collisionChecker = collisionChecker;
            m_movementPhysics = movementPhysics;
        }

        public void ResolveCollisions()
        {
            List<List<IPhysicalEntity>> collisionGroups = m_collisionChecker.CollisionGroups();
            foreach (List<IPhysicalEntity> collisionGroup in collisionGroups)
            {
                // TODO : resolve this unsafe cast before extending physical model
                ResolveCollision(collisionGroup.Cast<IForwardMovablePhysicalEntity>().ToList());
            }
        }

        private void ResolveCollision(List<IForwardMovablePhysicalEntity> collisionGroup)
        {
            MoveCollisionGroup(collisionGroup, -1);

            // if starting position is not permissible, make random walk
            foreach (IForwardMovablePhysicalEntity physicalEntity in collisionGroup)
            {
                if (m_collisionChecker.Collides(physicalEntity))
                {
                    SetSomePositionAround(physicalEntity);
                }
            }

            float timeLeft = 1;

            int counter = 0;
            float prevTimeLeft = -1;
            do
            {
                timeLeft = CollisionFreePositionBinarySearch(collisionGroup, timeLeft);
                SetNewSpeedsAndDirections(collisionGroup, timeLeft);
                counter++;
                /*if (Math.Abs(prevTimeLeft - timeLeft) < NEGLIGIBLE_TIME)
                {
                    break;
                }*/
                prevTimeLeft = timeLeft;
            } while (timeLeft - NEGLIGIBLE_TIME > 0 && counter < MAXIMUM_RESOLVE_TRIES);
        }

        private void SetNewSpeedsAndDirections(List<IForwardMovablePhysicalEntity> collisionGroup, float timeLeft)
        {
            // resolve collisions with tiles first
            foreach (IForwardMovablePhysicalEntity physicalEntity in collisionGroup)
            {
                /*var collidesWithTile = m_collisionChecker.CollidesWithTile(physicalEntity,
                    physicalEntity.ForwardSpeed / BINARY_SEARCH_ITERATIONS + NEGLIGIBLE_DISTANCE);

                if (collidesWithTile)
                {
                    FindTileFreeDirection(physicalEntity, timeLeft);
                }*/
            }

            // then set new directions WRT other objects
            for (int index = 0; index < collisionGroup.Count - 1; index++)
            {
                /*IForwardMovablePhysicalEntity physicalEntity = collisionGroup[index];
                List<IForwardMovablePhysicalEntity> threatenedObjects = m_collisionChecker.CollisionThreat(
                    physicalEntity,
                    collisionGroup.Skip(index).Cast<IPhysicalEntity>().ToList(),
                    physicalEntity.ForwardSpeed / BINARY_SEARCH_ITERATIONS + NEGLIGIBLE_DISTANCE)
                    .Cast<IForwardMovablePhysicalEntity>()
                    .ToList();

                if (threatenedObjects.Count > 0)
                {
                    threatenedObjects.ForEach(x => ResolveCollisionOfTwo(physicalEntity, x));
                }*/
            }
        }

        private void ResolveCollisionOfTwo(IForwardMovablePhysicalEntity pe0, IForwardMovablePhysicalEntity pe1)
        {
            if (pe0 == pe1)
            {
                Debug.Assert(false);
            }
            Vector2 diff = (pe0.Position - pe1.Position);
            Vector2 orthogonal = new Vector2(-diff.Y, diff.X);
            Vector2 normalOrthogonal = Vector2.Normalize(orthogonal);
            float orthogonalAngle0 = (float)Math.Atan2(normalOrthogonal.Y, normalOrthogonal.X);
            float orthogonalAngle1 = (float)Math.Atan2(- normalOrthogonal.Y, - normalOrthogonal.X);

            CollidesWithLine(pe0, orthogonalAngle0);
            CollidesWithLine(pe1, orthogonalAngle1);
        }

        private void CollidesWithLine(IForwardMovablePhysicalEntity target, float normAngle)
        {
            if (target.ElasticCollision)
            {
                if (Math.Abs(MathHelper.WrapAngle(normAngle - target.Direction)) <= MathHelper.Pi / 2)
                {
                    target.Direction = MathHelper.WrapAngle(2 * normAngle - MathHelper.Pi - target.Direction);
                }
                else
                {
                    target.Direction = MathHelper.WrapAngle(2 * normAngle - target.Direction);
                }
            }
            else if (target.InelasticCollision)
            {
                float cosWrtDirection0 = (float)Math.Sin(normAngle - target.Direction);
                target.ForwardSpeed *= cosWrtDirection0;
                target.Direction = normAngle;
            }
            else
            {
                target.ForwardSpeed = 0;
            }
        }

        private void FindTileFreeDirection(IForwardMovablePhysicalEntity physicalEntity, float timeLeft)
        {
            if (physicalEntity.ElasticCollision)
            {
                BounceFromTile(physicalEntity, timeLeft);
            }
            else if (physicalEntity.InelasticCollision)
            {
                SlideAroundTile(physicalEntity, timeLeft);
            }
            else
            {
                physicalEntity.ForwardSpeed = 0;
            }
        }

        private void BounceFromTile(IForwardMovablePhysicalEntity physicalEntity, float speed)
        {
            Vector2 originalPosition = physicalEntity.Position;
            float originalDirection = physicalEntity.Direction;

            float maxDistance = 0;
            float bestDirection = originalDirection;

            // candidate directions to move
            var candidateDirections = new List<float>
            {
                MathHelper.WrapAngle(2f*MathHelper.Pi - physicalEntity.Direction),
                MathHelper.WrapAngle(3f*MathHelper.Pi - physicalEntity.Direction)
            };

            // search for direction of longest move
            foreach (float newDirection in candidateDirections)
            {
                TileFreePositionBinarySearch(physicalEntity, speed, newDirection);

                float distance = Vector2.Distance(originalPosition, physicalEntity.Position);
                if (maxDistance < distance)
                {
                    maxDistance = distance;
                    bestDirection = newDirection;
                }

                physicalEntity.Position = originalPosition;
            }
            physicalEntity.Direction = bestDirection;
        }

        private void SlideAroundTile(IForwardMovablePhysicalEntity physicalEntity, float time)
        {
            Vector2 freePosition = physicalEntity.Position;
            float directionRads = physicalEntity.Direction;

            float speed = time*physicalEntity.ForwardSpeed;

            float xSpeed = (float)Math.Sin(directionRads) * speed;
            float ySpeed = (float)Math.Cos(directionRads) * speed;
            // position before move

            // try to move orthogonally left/right and up/down
            TileFreePositionBinarySearch(physicalEntity, xSpeed, X_DIRECTION);
            Vector2 xPosition = new Vector2(physicalEntity.Position.X, physicalEntity.Position.Y);
            physicalEntity.Position = freePosition;

            // try to move orthogonally up/down and left/right; reset position first
            TileFreePositionBinarySearch(physicalEntity, ySpeed, Y_DIRECTION);
            Vector2 yPosition = new Vector2(physicalEntity.Position.X, physicalEntity.Position.Y);
            physicalEntity.Position = freePosition;

            float distanceX = Vector2.Distance(freePosition, xPosition);
            float distanceY = Vector2.Distance(freePosition, yPosition);
            if (Math.Abs(distanceX) < NEGLIGIBLE_DISTANCE &&
                Math.Abs(distanceY) < NEGLIGIBLE_DISTANCE)
            {
                physicalEntity.ForwardSpeed = 0;
                return;
            }

            // farther position is chosen
            if (distanceX > distanceY)
            {
                Vector2 diff = xPosition - freePosition;
                physicalEntity.Direction = - (float)Math.Atan2(diff.X, diff.Y);
                physicalEntity.ForwardSpeed = Math.Abs(xSpeed) / time;
            }
            else
            {
                Vector2 diff = yPosition - freePosition;
                physicalEntity.Direction = (float)Math.Atan2(diff.X, diff.Y);
                physicalEntity.ForwardSpeed = Math.Abs(ySpeed) / time;
            }

            
        }

        /// <summary>
        /// Search for position close to obstacle in given direction. Must be on stable position when starting search.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <param name="initialSpeed"></param>
        /// <param name="direction"></param>
        private void TileFreePositionBinarySearch(IForwardMovablePhysicalEntity physicalEntity, float initialSpeed, float direction)
        {
            float speed = initialSpeed;
            
            Vector2 lastNotColliding = physicalEntity.Position;

            bool goForward = true;

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

        /// <summary>
        /// Search for position close to obstacle in given direction. Must be on stable position when starting search.
        /// </summary>
        /// <param name="collisionGroup"></param>
        /// <param name="initialTime"></param>
        /// <returns></returns>
        private float CollisionFreePositionBinarySearch(List<IForwardMovablePhysicalEntity> collisionGroup, float initialTime)
        {
            double time = initialTime;
            double timeLeft = initialTime;

            List<Vector2> lastNotCollidingPositions = GetPositions(collisionGroup);

            bool goForward = true;

            for (int i = 0; i < BINARY_SEARCH_ITERATIONS; i++)
            {
                if (goForward)
                {
                    MoveCollisionGroup(collisionGroup, (float)time);
                    timeLeft -= time;
                }
                else
                {
                    MoveCollisionGroup(collisionGroup, (float)-time);
                    timeLeft += time;
                }
                bool colliding = m_collisionChecker.Collides(collisionGroup.Cast<IPhysicalEntity>().ToList()) > 0;
                if (!colliding)
                {
                    lastNotCollidingPositions = GetPositions(collisionGroup);
                }
                if (timeLeft <= 0 && !colliding)
                {
                    break;
                }
                time /= 2;
                goForward = !colliding;
            }

            SetPositions(collisionGroup, lastNotCollidingPositions);

            return (float)timeLeft;
        }

        private void SetSomePositionAround(IPhysicalEntity physicalEntity)
        {
            // search in spirals
            float a = 0.1f;
            float tDiff = MathHelper.ToRadians(10);
            float t = 0;
            do
            {
                t += tDiff;
                physicalEntity.Position = new Vector2(
                    a*t*(float) Math.Cos(t) + physicalEntity.Position.X,
                    a*t*(float) Math.Sin(t) + physicalEntity.Position.Y);
            } while (m_collisionChecker.Collides(physicalEntity));
        }

        private static List<Vector2> GetPositions(List<IForwardMovablePhysicalEntity> collisionGroup)
        {
            return collisionGroup.Select(x => x.Position).ToList();
        }

        private static void SetPositions(List<IForwardMovablePhysicalEntity> collisionGroup, List<Vector2> positions)
        {
            for (int i = 0; i < collisionGroup.Count; i++)
            {
                collisionGroup[i].Position = positions[i];
            }
        }

        private void MoveCollisionGroup(IEnumerable<IForwardMovablePhysicalEntity> collisionGroup, float time)
        {
            foreach (IForwardMovablePhysicalEntity physicalEntity in collisionGroup)
            {
                m_movementPhysics.Shift(physicalEntity, time * physicalEntity.ForwardSpeed);
            }
        }
    }
}
