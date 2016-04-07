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
            var directionInRads = VRageMath.MathHelper.ToRadians(movable.Direction);
            var cos = (float)Math.Cos(directionInRads);
            float x = movable.Position.X + cos * movable.ForwardSpeed;
            var sin = (float)Math.Sin(directionInRads);
            float y = movable.Position.Y + sin * movable.ForwardSpeed;
            movable.Position = new Vector2(x, y);
        }

        private static void Rotate(IMovableDirectablePhysicalEntity movable)
        {
            movable.Direction += movable.RotationSpeed;
        }
    }
}
