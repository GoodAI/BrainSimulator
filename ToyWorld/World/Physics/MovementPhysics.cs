using System;
using VRageMath;

namespace World.Physics
{
    public interface IMovementPhysics
    {
        void Move(IForwardMovablePhysicalEntity movable);
    }

    public class MovementPhysics : IMovementPhysics
    {
        public void Move(IForwardMovablePhysicalEntity movable)
        {
            Shift(movable);
            Rotate(movable);
        }

        private static void Shift(IForwardMovablePhysicalEntity movable)
        {
            var directionInRads = VRageMath.MathHelper.ToRadians(movable.Direction);
            var cos = (float)Math.Cos(directionInRads);
            float x = movable.Position.X + cos * movable.ForwardSpeed;
            var sin = (float)Math.Sin(directionInRads);
            float y = movable.Position.Y + sin * movable.ForwardSpeed;
            movable.Position = new Vector2(x, y);
        }

        private static void Rotate(IForwardMovablePhysicalEntity movable)
        {
            movable.Direction += movable.RotationSpeed;
        }
    }
}
