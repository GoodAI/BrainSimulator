using System.Collections.Generic;
using System.Diagnostics;
using VRageMath;

namespace World.Physics
{
    public class StaticPhysicalEntity : IPhysicalEntity
    {
        private Vector2 m_position;

        public Vector2 Position
        {
            get { return m_position; }
            set
            {
                Debug.Assert(false, "StaticPhysicalEntity Position setter used");
                m_position = value;
            }
        }

        public IShape Shape { get; set; }
        public bool InelasticCollision { get; set; }
        public bool ElasticCollision { get; set; }
        public float Weight { get; set; }

        public StaticPhysicalEntity(IShape shape, float weight = 1, bool inelasticCollision = false, bool elasticCollision = false)
        {
            Shape = shape;
            InelasticCollision = inelasticCollision;
            ElasticCollision = elasticCollision;
            Weight = weight;
        }

        public RectangleF CoverRectangle()
        {
            throw new System.NotImplementedException();
        }

        public List<Vector2I> CoverTiles()
        {
            throw new System.NotImplementedException();
        }

        public bool CollidesWith(IPhysicalEntity physicalEntity)
        {
            return Shape.CollidesWith(physicalEntity.Shape);
        }
    }
}