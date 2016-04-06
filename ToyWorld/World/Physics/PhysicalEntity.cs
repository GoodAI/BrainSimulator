using VRageMath;

namespace World.Physics
{
    public interface IPhysicalEntity
    {
        /// <summary>
        /// Absolute position in ToyWorld.
        /// </summary>
        Vector2 Position { get; set; }
    }

    public abstract class PhysicalEntity : IPhysicalEntity
    {
        /// <summary>
        /// Absolute position in ToyWorld.
        /// </summary>
        public Vector2 Position { get; set; }

        protected PhysicalEntity(Vector2 position)
        {
            Position = position;
        }
    }
}
