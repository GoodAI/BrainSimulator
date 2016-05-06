using VRageMath;

namespace World.Physics
{
    public interface IMovementPhysics
    {
        void Move(IForwardMovablePhysicalEntity movable);
        void Shift(IForwardMovablePhysicalEntity movable);
        void Shift(IForwardMovablePhysicalEntity movable, float speed);
        void Shift(IForwardMovablePhysicalEntity movable, float speed, float directionInRads);
    }

    public class MovementPhysics : IMovementPhysics
    {
        public void Move(IForwardMovablePhysicalEntity movable)
        {
            Rotate(movable);
            Shift(movable);
        }

        public void RevertMove(IForwardMovablePhysicalEntity movable)
        {
            Shift(movable, -movable.ForwardSpeed);
            Rotate(movable, -movable.RotationSpeed);
        }

        public void RevertMoveKeepRotation(IForwardMovablePhysicalEntity movable)
        {
            Shift(movable, -movable.ForwardSpeed);
        }

        public void Shift(IForwardMovablePhysicalEntity movable)
        {
            movable.Position = Utils.Move(movable.Position, movable.Direction, movable.ForwardSpeed);
        }

        public void Shift(IForwardMovablePhysicalEntity movable, float speed)
        {
            movable.Position = Utils.Move(movable.Position, movable.Direction, speed);
        }

        public void Shift(IForwardMovablePhysicalEntity movable, float speed, float directionInRads)
        {
            movable.Position = Utils.Move(movable.Position, directionInRads, speed);
        }

        private static void Rotate(IForwardMovablePhysicalEntity movable)
        {
            movable.Direction = MathHelper.WrapAngle(movable.Direction + movable.RotationSpeed);
        }

        private static void Rotate(IForwardMovablePhysicalEntity movable, float rotationSpeed)
        {
            movable.Direction = MathHelper.WrapAngle(movable.Direction + rotationSpeed);
        }
    }
}
