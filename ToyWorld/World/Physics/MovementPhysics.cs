using System;
using VRageMath;

namespace World.Physics
{
    public class MovementPhysics
    {
        public static void Move(IMovableDirectablePhysicalEntity movable)
        {
            Shift(movable);
            Rotate(movable);
        }

        private static void Shift(IMovableDirectablePhysicalEntity movable)
        {
            var directionInRads = movable.Direction / (180 / (float)Math.PI);
            float x = movable.Position.X + (float)Math.Cos(directionInRads) * movable.ForwardSpeed;
            float y = movable.Position.Y + (float)Math.Sin(directionInRads) * movable.ForwardSpeed;
            movable.Position = new Vector2(x, y);
        }

        private static void Rotate(IMovableDirectablePhysicalEntity movable)
        {
            movable.Direction += movable.RotationSpeed;
        }
    }
}
