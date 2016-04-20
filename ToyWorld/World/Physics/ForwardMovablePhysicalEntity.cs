using VRageMath;

namespace World.Physics
{
    public interface IForwardMovablePhysicalEntity : IForwardMovable, IPhysicalEntity
    {
    }

    public class ForwardMovablePhysicalEntity : PhysicalEntity, IForwardMovablePhysicalEntity
    {
        public float Direction { get; set; }
        public float ForwardSpeed { get; set; }
        public float RotationSpeed { get; set; }

        public ForwardMovablePhysicalEntity(
            Vector2 initialPostition,
            Shape shape,
            float forwardSpeed = 0,
            float direction = 90,
            float rotationSpeed = 0)
            : base(initialPostition, shape)
        {
            ForwardSpeed = forwardSpeed;
            Direction = direction;
            RotationSpeed = rotationSpeed;
        }
    }
}
