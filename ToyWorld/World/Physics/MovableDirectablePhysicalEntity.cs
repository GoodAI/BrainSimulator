using GoodAI.ToyWorld.Control;
using VRageMath;

namespace World.Physics
{
    public interface IMovableDirectablePhysicalEntity
    {
        float Direction { get; set; }
        float ForwardSpeed { get; set; }
        float RotationSpeed { get; set; }

        /// <summary>
        /// Absolute position in ToyWorld.
        /// </summary>
        Vector2 Position { get; set; }
    }

    public class MovableDirectablePhysicalEntity : PhysicalEntity, IDirectable, IForwardMovable, IMovableDirectablePhysicalEntity
    {
        public float Direction { get; set; }
        public float ForwardSpeed { get; set; }
        public float RotationSpeed { get; set; }

        public MovableDirectablePhysicalEntity(
            Vector2 initialPostition,
            float forwardSpeed = 0,
            float direction = 90,
            float rotationSpeed = 0) : base(initialPostition)
        {
            ForwardSpeed = forwardSpeed;
            Direction = direction;
            RotationSpeed = rotationSpeed;
        }
    }
}
