using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using GoodAI.Logging;
using VRageMath;

namespace World.Physics
{
    public class MomentumCollisionResolver : ICollisionResolver
    {
        private readonly ICollisionChecker m_collisionChecker;
        private readonly IMovementPhysics m_movementPhysics;

        private const int BINARY_SEARCH_ITERATIONS = 14;
        private const float X_DIRECTION = (float) Math.PI/2;
        private const float Y_DIRECTION = 0;
        private const float NEGLIGIBLE_TIME = 0.01f;
        private const float NEGLIGIBLE_DISTANCE = 0.0001f;
        private const int MAXIMUM_RESOLVE_TRIES = 25;

        public MomentumCollisionResolver(ICollisionChecker collisionChecker, IMovementPhysics movementPhysics)
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
                ResolveCollision(collisionGroup);
            }
        }

        private void ResolveCollision(List<IPhysicalEntity> collisionGroup)
        {
            // move back to safe position
            MoveCollisionGroup(collisionGroup, -1);

            float remainingTime = 1.0f;
            int counter = 0;
            do
            {
                // if starting position is not permissible, make random walk
                foreach (IPhysicalEntity physicalEntity in collisionGroup)
                {
                    if (m_collisionChecker.Collides(physicalEntity))
                    {
                        SetSomePositionAround(physicalEntity);
                    }
                }

                Tuple<float, float> res = CollisionFreePositionBinarySearch(collisionGroup, remainingTime);
                var lastNotCollidingTime = res.Item1;
                var firstCollidingTime = res.Item2;

                if (float.IsPositiveInfinity(firstCollidingTime))
                {
                    MoveCollisionGroup(collisionGroup, remainingTime);
                }
                else
                {
                    SetNewSpeedsAndDirectionsAndMoveBeforeCollision(collisionGroup, lastNotCollidingTime,
                        firstCollidingTime);
                }

                counter++;
                remainingTime = remainingTime - lastNotCollidingTime;
            } while (remainingTime - NEGLIGIBLE_TIME > 0.0f && counter < MAXIMUM_RESOLVE_TRIES);

            if (counter >= MAXIMUM_RESOLVE_TRIES)
            {
                // moves the system into a collision state
                // will force calling SetSomePositionAround(physicalEntity); in next step
                MoveCollisionGroup(collisionGroup, remainingTime);
            }
        }

        private void SetNewSpeedsAndDirectionsAndMoveBeforeCollision(List<IPhysicalEntity> collisionGroup,
            float lastNotCollidingTime, float firstCollidingTime)
        {
            // positions in first collision time.
            List<Vector2> collisionPositions;
            var collisionPairs = ComputeCollisionPairs(out collisionPositions, collisionGroup, firstCollidingTime);

            MoveCollisionGroup(collisionGroup, lastNotCollidingTime);

            collisionPairs.ForEach(ResolveCollisionPair);

            // resolve collisions with tiles
            ResolveCollisionsWithTiles(collisionGroup, collisionPositions);
        }

        private void ResolveCollisionPair(Tuple<IPhysicalEntity, IPhysicalEntity> collisionPair)
        {
            ResolveCollisionOfTwoIForwardMovablePe((IForwardMovablePhysicalEntity)collisionPair.Item1, (IForwardMovablePhysicalEntity)collisionPair.Item2);
        }

        private void ResolveCollisionsWithTiles(List<IPhysicalEntity> collisionGroup, List<Vector2> collisionPositions)
        {
            for (int i = 0; i < collisionGroup.Count; i++)
            {
                IPhysicalEntity physicalEntity = collisionGroup[i];
                Vector2 origPos = physicalEntity.Position;
                physicalEntity.Position = collisionPositions[i];
                var collidesWithTile = m_collisionChecker.CollidesWithTile(physicalEntity);
                physicalEntity.Position = origPos;

                if (collidesWithTile)
                {
                    ResolveTileCollision((IForwardMovablePhysicalEntity)physicalEntity);
                }
            }
        }

        private List<Tuple<IPhysicalEntity, IPhysicalEntity>> ComputeCollisionPairs(
            out List<Vector2> collisionPositions, List<IPhysicalEntity> collisionGroup,
            float firstCollidingTime)
        {
            // make a minimal step forward. There is a collision in the next minimal time step - resolve that collision
            List<Vector2> positions = GetPositions(collisionGroup);

            MoveCollisionGroup(collisionGroup, firstCollidingTime);

            collisionPositions = GetPositions(collisionGroup);

            List<IPhysicalEntity> objectsInCollision =
                collisionGroup.Where(m_collisionChecker.Collides).ToList();

            List<Tuple<IPhysicalEntity, IPhysicalEntity>> collidingCouples =
                CollisionChecker.CollidingCouples(objectsInCollision);

            SetPositions(collisionGroup, positions);
            return collidingCouples;
        }

        private void ResolveCollisionOfTwoIForwardMovablePe(IForwardMovablePhysicalEntity pe0, IForwardMovablePhysicalEntity pe1)
        {
            if (pe0 == pe1)
            {
                return;
            }
            Vector2 diff = (pe0.Position - pe1.Position);
            Vector2 normalDiff = Vector2.Normalize(diff);

            float diffAngle = (float)Math.Atan2(normalDiff.X, -normalDiff.Y);

            if (pe0.ElasticCollision || pe1.ElasticCollision)
            {
                ElasticCollision(pe0, pe1, diffAngle);
            }
            else if (pe0.InelasticCollision || pe1.InelasticCollision)
            {
                if (pe0.InelasticCollision)
                {
                    InelasticCollision(pe0, pe1, diffAngle);
                }
            }
            else
            {
                pe0.ForwardSpeed = 0;
            }
        }

        private void InelasticCollision(IForwardMovablePhysicalEntity target, IForwardMovablePhysicalEntity source,
            float diffAngle)
        {
            Vector2 decomposedSpeedTarget = Utils.DecomposeSpeed(target.ForwardSpeed, target.Direction, diffAngle);
            Vector2 decomposedSpeedSource = Utils.DecomposeSpeed(source.ForwardSpeed, source.Direction, diffAngle);

            float v1 = decomposedSpeedTarget.Y;
            float v2 = decomposedSpeedSource.Y;

            float m1 = target.Weight;
            float m2 = source.Weight;

            float newSpeedTargetAndSource = (m1*v1 + m2*v2)/(m1 + m2);

            decomposedSpeedTarget.Y = newSpeedTargetAndSource;
            decomposedSpeedSource.Y = newSpeedTargetAndSource;

            Tuple<float, float> composedSpeedTarget = Utils.ComposeSpeed(decomposedSpeedTarget, diffAngle);
            Tuple<float, float> composedSpeedSource = Utils.ComposeSpeed(decomposedSpeedSource, diffAngle);

            float targetSpeed = composedSpeedTarget.Item1;
            float sourceSpeed = composedSpeedSource.Item1;

            target.ForwardSpeed = targetSpeed > m_collisionChecker.MaximumGameObjectSpeed
                ? m_collisionChecker.MaximumGameObjectSpeed
                : targetSpeed;
            target.Direction = composedSpeedTarget.Item2;
            source.ForwardSpeed = sourceSpeed > m_collisionChecker.MaximumGameObjectSpeed
                ? m_collisionChecker.MaximumGameObjectSpeed
                : sourceSpeed;
            source.Direction = composedSpeedSource.Item2;
        }

        private void ElasticCollision(IForwardMovablePhysicalEntity target, IForwardMovablePhysicalEntity source,
            float diffAngle)
        {
            Vector2 decomposedSpeedTarget = Utils.DecomposeSpeed(target.ForwardSpeed, target.Direction, diffAngle);
            Vector2 decomposedSpeedSource = Utils.DecomposeSpeed(source.ForwardSpeed, source.Direction, diffAngle);
            float v1 = decomposedSpeedTarget.Y;
            float v2 = decomposedSpeedSource.Y;

            float m1 = target.Weight;
            float m2 = source.Weight;

            float originalEnergy = target.ForwardSpeed*target.ForwardSpeed*m1 +
                                   source.ForwardSpeed*source.ForwardSpeed*m2;

            double p = m1*v1 + m2*v2; // P = momentum
            double d = 4.0*m1*m1*m2*m2*(v1 - v2)*(v1 - v2); // D = determinant
            float v2A = (float) ((2*p*m2 + Math.Sqrt(d))/(2*(m1*m2 + m2*m2)));
            float v2B = (float) ((2*p*m2 - Math.Sqrt(d))/(2*(m1*m2 + m2*m2)));

            float newSourceSpeed = v2A;
            if (Math.Abs(v2A - v2) < Math.Abs(v2B - v2)) // one of the results is identity
                newSourceSpeed = v2B;

            float newTargetSpeed = (m1*v1 + m2*v2 - m2*newSourceSpeed)/m1;

            /* 
                float E2C = v1 * v1 * m1 + v2 * v2 * m2;
                float E2D = newTargetSpeed * newTargetSpeed * m1 + newSourceSpeed * newSourceSpeed * m2;
                Debug.Assert(Math.Abs(E2C - E2D) < 0.000001); // energy conservation check */

            decomposedSpeedTarget.Y = newTargetSpeed;
            decomposedSpeedSource.Y = newSourceSpeed;
            Tuple<float, float> composedSpeedTarget = Utils.ComposeSpeed(decomposedSpeedTarget, diffAngle);
            Tuple<float, float> composedSpeedSource = Utils.ComposeSpeed(decomposedSpeedSource, diffAngle);

            float targetSpeed = composedSpeedTarget.Item1;
            float sourceSpeed = composedSpeedSource.Item1;
            float finalEnergy = targetSpeed*targetSpeed*m1 + sourceSpeed*sourceSpeed*m2;

            Debug.Assert(Math.Abs(originalEnergy - finalEnergy) < 0.01);

            target.ForwardSpeed = targetSpeed > m_collisionChecker.MaximumGameObjectSpeed
                ? m_collisionChecker.MaximumGameObjectSpeed
                : targetSpeed;
            target.Direction = composedSpeedTarget.Item2;
            source.ForwardSpeed = sourceSpeed > m_collisionChecker.MaximumGameObjectSpeed
                ? m_collisionChecker.MaximumGameObjectSpeed
                : sourceSpeed;
            source.Direction = composedSpeedSource.Item2;
        }

        private void ResolveTileCollision(IForwardMovablePhysicalEntity physicalEntity)
        {
            float timeStep = 1.0f;

            if (physicalEntity.ElasticCollision)
            {
                BounceFromTile(physicalEntity, timeStep);
            }
            else if (physicalEntity.InelasticCollision)
            {
                SlideAroundTile(physicalEntity, timeStep);
            }
            else
            {
                physicalEntity.ForwardSpeed = 0;
            }
        }

        private void BounceFromTile(IForwardMovablePhysicalEntity physicalEntity, float speed)
        {
            TileFreePositionBinarySearch(physicalEntity, physicalEntity.ForwardSpeed, physicalEntity.Direction);

            Vector2 originalPosition = physicalEntity.Position;
            float originalDirection = physicalEntity.Direction;

            float maxDistance = 0;
            float bestDirection = MathHelper.WrapAngle(originalDirection + (float) Math.PI);
                // emergency reverse direction if either bounce fails

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

            float xSpeed = (float) Math.Sin(directionRads)*speed;
            float ySpeed = (float) Math.Cos(directionRads)*speed;
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
                physicalEntity.Direction = -(float) Math.Atan2(diff.X, diff.Y);
                physicalEntity.ForwardSpeed = Math.Abs(xSpeed)/time;
            }
            else
            {
                Vector2 diff = yPosition - freePosition;
                physicalEntity.Direction = (float) Math.Atan2(diff.X, diff.Y);
                physicalEntity.ForwardSpeed = Math.Abs(ySpeed)/time;
            }
        }

        /// <summary>
        /// Search for position close to obstacle in given direction. Must be on stable position when starting search.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <param name="initialSpeed"></param>
        /// <param name="direction"></param>
        private void TileFreePositionBinarySearch(IForwardMovablePhysicalEntity physicalEntity, float initialSpeed,
            float direction)
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
                speed = speed/2;
                goForward = !colliding;
            }

            physicalEntity.Position = lastNotColliding;
        }

        /// <summary>
        /// Search for position of all objects before and after first collision. Must be on safe position when starting search.
        /// </summary>
        /// <param name="collisionGroup"></param>
        /// <param name="initialTime"></param>
        /// <param name="maxCollidingCouples"></param>
        /// <returns>lastNotCollidingTime, firstCollidingTime (can be positive infinity)</returns>
        private Tuple<float, float> CollisionFreePositionBinarySearch(
            List<IPhysicalEntity> collisionGroup, float initialTime, int maxCollidingCouples = 0)
        {
            double timeStep = initialTime;
            double time = 0;
            double lastNotCollidingTime = 0.0f;
            double firstCollidingTime = double.PositiveInfinity;

            List<Vector2> initialPositions = GetPositions(collisionGroup);

            Debug.Assert(m_collisionChecker.Collides(collisionGroup.ToList()) == 0);

            bool goForward = true;

            for (int i = 0; i < BINARY_SEARCH_ITERATIONS; i++)
            {
                SetPositions(collisionGroup, initialPositions);

                if (goForward)
                {
                    time += timeStep;
                }
                else
                {
                    time -= timeStep;
                }
                MoveCollisionGroup(collisionGroup, (float) time);

                int couplesInCollision = m_collisionChecker.Collides(collisionGroup.ToList());
                bool colliding = couplesInCollision > maxCollidingCouples;
                if (!colliding)
                {
                    lastNotCollidingTime = time;
                }
                else
                {
                    firstCollidingTime = time;
                }
                if (time >= 1 && !colliding)
                {
                    break;
                }
                timeStep /= 2;
                goForward = !colliding;
            }

            SetPositions(collisionGroup, initialPositions);

            return new Tuple<float, float>((float) lastNotCollidingTime, (float) firstCollidingTime);
        }

        private void SetSomePositionAround(IPhysicalEntity physicalEntity)
        {
            Vector2 originalPosition = physicalEntity.Position;
            // search in spirals
            const float a = 0.01f;
            const float tDiff = MathHelper.Pi / 180; // one degree step
            float t = 0;
            do
            {
                t += tDiff;
                physicalEntity.Position = new Vector2(
                    a*t*(float) Math.Cos(t) + physicalEntity.Position.X,
                    a*t*(float) Math.Sin(t) + physicalEntity.Position.Y);
            } while (m_collisionChecker.Collides(physicalEntity) && t < 1000);

            if (t >= 1000)
            {
                physicalEntity.Position = originalPosition;
                Log.Instance.Warn(GetType().Name + " cannot place one of objects on position " + physicalEntity.Position + ".");
            }
        }

        private static List<Vector2> GetPositions(IEnumerable<IPhysicalEntity> collisionGroup)
        {
            return collisionGroup.Select(x => x.Position).ToList();
        }

        private static void SetPositions(List<IPhysicalEntity> collisionGroup, List<Vector2> positions)
        {
            for (int i = 0; i < collisionGroup.Count; i++)
            {
                collisionGroup[i].Position = positions[i];
            }
        }

        /// <summary>
        /// Move whole collision group in time according to objects speeds and directions.
        /// </summary>
        /// <param name="collisionGroup"></param>
        /// <param name="time"></param>
        private void MoveCollisionGroup(IEnumerable<IPhysicalEntity> collisionGroup, float time)
        {
            Debug.Assert(!float.IsNaN(time) && !float.IsInfinity(time));
            foreach (var physicalEntity1 in collisionGroup)
            {
                var physicalEntity = (IForwardMovablePhysicalEntity) physicalEntity1;
                m_movementPhysics.Shift(physicalEntity, time*physicalEntity.ForwardSpeed);
            }
        }
    }
}
