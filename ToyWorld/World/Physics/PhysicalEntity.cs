using System.Collections.Generic;
using VRageMath;

namespace World.Physics
{
    public interface IPhysicalEntity
    {
        /// <summary>
        /// Absolute position in ToyWorld. Upper left corner.
        /// </summary>
        Vector2 Position { get; set; }

        /// <summary>
        /// Shape of this entity.
        /// </summary>
        IShape Shape { get; set; }

        /// <summary>
        /// Returns rectangle wrapping this entity.
        /// </summary>
        /// <returns></returns>
        RectangleF CoverRectangle();

        /// <summary>
        /// Returns list of all square coordinates which collides with this entity.
        /// </summary>
        /// <returns></returns>
        List<Vector2I> CoverTiles();
    }

    public abstract class PhysicalEntity : IPhysicalEntity
    {
        public Vector2 Position {
            get { return Shape.Position; }
            set { Shape.Position = value; }
        }

        public IShape Shape { get; set; }

        public PhysicalEntity(Vector2 position, Shape shape)
        {
            Shape = shape;
            Position = position;
        }

        public VRageMath.RectangleF CoverRectangle()
        {
            return new VRageMath.RectangleF(Position, Shape.CoverRectangleSize());
        }

        public List<Vector2I> CoverTiles()
        {
            return Shape.CoverTiles();
        }
    }
}
