
namespace World.Physics
{
    public interface IMovementPhysics
    {
        /// <summary>
        /// Move with given ForwardMovable object.
        /// </summary>
        /// <param name="movable"></param>
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
            movable.Move();
        }

        private static void Rotate(IForwardMovablePhysicalEntity movable)
        {
            movable.Direction += movable.RotationSpeed;
        }
    }
}
