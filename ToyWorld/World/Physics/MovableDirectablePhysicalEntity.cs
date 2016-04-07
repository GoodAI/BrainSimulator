using GoodAI.ToyWorld.Control;
using VRageMath;

namespace World.Physics
{
    public interface IMovableDirectablePhysicalEntity : IDirectable, IForwardMovable, IPhysicalEntity
    {
    }

    public class MovableDirectablePhysicalEntity : PhysicalEntity, IMovableDirectablePhysicalEntity
    {
        public float Direction { get; set; }
        public float ForwardSpeed { get; set; }
        public float RotationSpeed { get; set; }

        public MovableDirectablePhysicalEntity(
            Vector2 initialPostition,
            Vector2 size,
            float forwardSpeed = 0,
            float direction = 90,
            float rotationSpeed = 0) : base(initialPostition, size)
        {
            ForwardSpeed = forwardSpeed;
            Direction = direction;
            RotationSpeed = rotationSpeed;
        }
    }
}
