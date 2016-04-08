using VRageMath;

namespace World.Physics
{
    public interface IPhysicalEntity
    {
        /// <summary>
        /// Absolute position in ToyWorld.
        /// </summary>
        Vector2 Position { get; set; }
        Vector2 Size { get; set; }
    }

    public abstract class PhysicalEntity : IPhysicalEntity
    {
        /// <summary>
        /// Absolute position in ToyWorld.
        /// </summary>
        public Vector2 Position { get; set; }

        public Vector2 Size { get; set; }

        public PhysicalEntity(Vector2 position, Vector2 size)
        {
            Position = position;
            Size = size;
        }
    }
}
