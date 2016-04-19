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
            // Swapped sin and cos (rotated by 90°) because the default direction vector (for radian = 0) should be up (i.e. (0,1))
            Vector2 direction = new Vector2((float)Math.Sin(movable.Direction), (float)Math.Cos(movable.Direction));
            movable.Position = movable.Position + direction * movable.ForwardSpeed;
        }

        private static void Rotate(IForwardMovablePhysicalEntity movable)
        {
            movable.Direction = MathHelper.WrapAngle(movable.Direction + movable.RotationSpeed);
        }
    }
}
