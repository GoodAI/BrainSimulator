using System.Collections.Generic;
using VRageMath;
using World.ToyWorldCore;

namespace World.Physics
{
    public interface ICollisionChecker
    {
        /// <summary>
        /// Checks whether given PhysicalEntity collides with any other PhysicalEntity.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <returns>List of PhysicalEntities given entity collides with.</returns>
        List<PhysicalEntity> CollidesWithAnotherEntity(IPhysicalEntity physicalEntity);

        /// <summary>
        /// Checks whether given PhysicalEntity collides with any Tile.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <returns></returns>
        bool CollidesWithTile(IPhysicalEntity physicalEntity);

        /// <summary>
        /// Maximum size of an object. It is used in collisions check between PhysicalEntities.
        /// </summary>
        float MaximumGameObjectRadius { get; }
    }

    class CollisionChecker : ICollisionChecker
    {
        private const float MAXIMUM_GAMEOBJECT_RADIUS = 5f;
        public float MaximumGameObjectRadius { get { return MAXIMUM_GAMEOBJECT_RADIUS; } }

        private IAtlas m_atlas;

        public CollisionChecker(IAtlas atlas){
            m_atlas = atlas;
        }

        public bool CollidesWithTile(IPhysicalEntity physicalEntity){
            List<Vector2I> coverTilesCoordinates = physicalEntity.CoverTiles();
            bool colliding = !coverTilesCoordinates.TrueForAll(x => !m_atlas.ContainsCollidingTile(x));
            return colliding;
        }

        public List<PhysicalEntity> CollidesWithAnotherEntity(IPhysicalEntity physicalEntity){
            // TODO
            return null;
        }
    }
}
