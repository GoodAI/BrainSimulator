using System;
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

        bool InelasticCollision { get; set; }

        bool ElasticCollision { get; set; }

        bool StopOnCollision { get; set; }

        /// <summary>
        /// Weight of an object.
        /// </summary>
        float Weight { get; set; }

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

        /// <summary>
        /// Check for collision between two objects.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <param name="eps">Border width of this object.</param>
        /// <returns></returns>
        bool CollidesWith(IPhysicalEntity physicalEntity, float eps);

        /// <summary>
        /// Check for collision between two objects.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <returns></returns>
        bool CollidesWith(IPhysicalEntity physicalEntity);
    }

    public abstract class PhysicalEntity : IPhysicalEntity
    {
        private bool m_inelasticCollision;
        private bool m_elasticCollision;
        private bool m_stopOnCollision;
        private float m_weight = 1.0f;

        public float Weight
        {
            get { return m_weight; }
            set { m_weight = value; }
        }

        public Vector2 Position {
            get { return Shape.Position; }
            set { Shape.Position = value; }
        }

        public IShape Shape { get; set; }

        public bool InelasticCollision
        {
            get { return m_inelasticCollision; }
            set
            {
                if (value && (ElasticCollision || StopOnCollision))
                {
                    throw new ArgumentException(
                        "InelasticCollision property cannot be true if ElasticCollision or StopOnCollision is true.");
                }
                m_inelasticCollision = value;
            }
        }

        public bool ElasticCollision
        {
            get { return m_elasticCollision; }
            set
            {
                if (value && (InelasticCollision || StopOnCollision))
                {
                    throw new ArgumentException(
                        "ElasticCollision property cannot be true if InelasticCollision or StopOnCollision is true.");
                }
                m_elasticCollision = value;
            }
        }

        public bool StopOnCollision
        {
            get { return m_stopOnCollision; }
            set
            {
                if (value && (ElasticCollision || InelasticCollision))
                {
                    throw new ArgumentException(
                        "StopOnCollision property cannot be true if InelasticCollision or ElasticCollision is true.");
                }
                m_stopOnCollision = value;
            }
        }

        public PhysicalEntity(Vector2 position, Shape shape)
        {
            Shape = shape;
            Position = position;
        }

        public RectangleF CoverRectangle()
        {
            return new RectangleF(Position, Shape.CoverRectangleSize());
        }

        public List<Vector2I> CoverTiles()
        {
            return Shape.CoverTiles();
        }

        public bool CollidesWith(IPhysicalEntity physicalEntity, float eps)
        {
            Shape.Resize(eps);
            var collidesWith = Shape.CollidesWith(physicalEntity.Shape);
            Shape.Resize(-eps);
            return collidesWith;
        }


        public bool CollidesWith(IPhysicalEntity physicalEntity)
        {
            if (physicalEntity == this)
            {
                return false;
            }
            var collidesWith = Shape.CollidesWith(physicalEntity.Shape);
            return collidesWith;
        }
    }
}
