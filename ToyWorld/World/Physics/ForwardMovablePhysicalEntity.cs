using VRageMath;

namespace World.Physics
{
    public interface IForwardMovablePhysicalEntity : IForwardMovable, IPhysicalEntity
    {
        TileCollision TileCollision { get; }
    }

    public enum TileCollision
    {
        Slide = 0,
        Bounce = 1,
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
            float direction = 0,
            float rotationSpeed = 0,
            TileCollision tileCollision = TileCollision.Slide)
            : base(initialPostition, shape)
        {
            ForwardSpeed = forwardSpeed;
            Direction = direction;
            RotationSpeed = rotationSpeed;
            TileCollision = tileCollision;
        }

        public TileCollision TileCollision { get; private set; }
    }
}
