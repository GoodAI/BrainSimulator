using System;
using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.Atlas;
using World.Atlas.Layers;
using World.ToyWorldCore;

namespace World.Physics
{
    public interface ICollisionChecker
    {
        /// <summary>
        /// Checks whether given PhysicalEntity collides with any other PhysicalEntity.
        /// </summary>
        /// <returns>List of PhysicalEntities given entity collides with.</returns>
        List<List<IPhysicalEntity>> CollisionGroups();

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

        /// <summary>
        /// Maximum speed of object. It is used in collisions check between PhysicalEntities.
        /// </summary>
        float MaximumGameObjectSpeed { get; }

        /// <summary>
        /// Check whether this physical entities collide with each other or tile.
        /// </summary>
        /// <param name="physicalEntities"></param>
        /// <returns>Number of couples of colliding objects.</returns>
        int Collides(List<IPhysicalEntity> physicalEntities);

        /// <summary>
        /// Check whether this physical entity collides with another PE or tile.
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <returns></returns>
        bool Collides(IPhysicalEntity physicalEntity);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="physicalEntity"></param>
        /// <returns></returns>
        bool CollidesWithPhysicalEntity(IPhysicalEntity physicalEntity);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="physicalEntities"></param>
        /// <returns></returns>
        int NumberOfCollidingCouples(List<IPhysicalEntity> physicalEntities);
    }

    class CollisionChecker : ICollisionChecker
    {
        private readonly List<IPhysicalEntity> m_physicalEntities;
        private readonly IAtlas m_atlas;
        private readonly IObjectLayer m_objectLayer;

        private const float MAXIMUM_GAMEOBJECT_RADIUS = 5f;
        public float MaximumGameObjectRadius { get { return MAXIMUM_GAMEOBJECT_RADIUS; } }

        private const float MAXIMUM_OBJECT_SPEED = 1f;
        public float MaximumGameObjectSpeed { get { return MAXIMUM_OBJECT_SPEED; } }


        public CollisionChecker(IAtlas atlas)
        {
            m_atlas = atlas;
            m_objectLayer = atlas.GetLayer(LayerType.Object) as SimpleObjectLayer;
            if (m_objectLayer != null)
            {
                m_physicalEntities = m_objectLayer.GetPhysicalEntities();
            }
            else
            {
                throw new ArgumentException("ObjectLayer not found.");
            }
        }


        /// <summary>
        /// Search for all objects that are or can be in collision with each other.
        /// Must be called when step was already performed, so collisions are already there.
        /// If there is no collision, returns empty list.
        /// </summary>
        /// <returns>List of list of all potentially colliding objects.</returns>
        public List<List<IPhysicalEntity>> CollisionGroups()
        {
            List<HashSet<IPhysicalEntity>> listOfSets = new List<HashSet<IPhysicalEntity>>();

            foreach (IPhysicalEntity physicalEntity in m_physicalEntities)
            {
                if (Collides(physicalEntity))
                {
                    var circle = new Circle(physicalEntity.Position,
                        2 * MaximumGameObjectRadius + MaximumGameObjectSpeed);
                    var physicalEntities = new HashSet<IPhysicalEntity>(m_objectLayer.GetPhysicalEntities(circle));
                    listOfSets.Add(physicalEntities);
                }
            }

            // consolidation
            for (int i = 0; i < listOfSets.Count - 1; i++)
            {
                HashSet<IPhysicalEntity> firstList = listOfSets[i];
                for (int j = i + 1; j < listOfSets.Count; j++)
                {
                    HashSet<IPhysicalEntity> secondList = listOfSets[j];
                    if (firstList.Overlaps(secondList))
                    {
                        firstList.UnionWith(secondList);
                        listOfSets.RemoveAt(j);
                        j--;
                    }
                }
            }

            // To List
            List<List<IPhysicalEntity>> collisionGroups = new List<List<IPhysicalEntity>>();
            foreach (HashSet<IPhysicalEntity> hashSet in listOfSets)
            {
                collisionGroups.Add(hashSet.ToList());
            }

            return collisionGroups;
        }

        public int Collides(List<IPhysicalEntity> collisionGroup)
        {
            return collisionGroup.Count(CollidesWithTile) + NumberOfCollidingCouples(collisionGroup);
        }

        public bool Collides(IPhysicalEntity physicalEntity)
        {
            return CollidesWithTile(physicalEntity) || CollidesWithPhysicalEntity(physicalEntity);
        }

        public bool CollidesWithTile(IPhysicalEntity physicalEntity)
        {
            if (!physicalEntity.ElasticCollision && !physicalEntity.InelasticCollision) return false;
            List<Vector2I> coverTilesCoordinates = physicalEntity.CoverTiles();
            bool colliding = !coverTilesCoordinates.TrueForAll(x => !m_atlas.ContainsCollidingTile(x));
            return colliding;
        }

        public bool CollidesWithPhysicalEntity(IPhysicalEntity physicalEntity)
        {
            if (!physicalEntity.ElasticCollision && !physicalEntity.InelasticCollision) return false;
            var circle = new Circle(physicalEntity.Position, 2 * MaximumGameObjectRadius);
            List<IPhysicalEntity> physicalEntities = m_objectLayer.GetPhysicalEntities(circle);
            return physicalEntities.Any(physicalEntity.CollidesWith);
        }

        public int NumberOfCollidingCouples(List<IPhysicalEntity> physicalEntities)
        {
            int counter = 0;
            for (int i = 0; i < physicalEntities.Count - 1; i++)
            {
                for (int j = i + 1; j < physicalEntities.Count; j++)
                {
                    if (physicalEntities[i].CollidesWith(physicalEntities[j]))
                    {
                        counter++;
                    }
                }
            }
            return counter;
        }

        public static List<Tuple<IPhysicalEntity, IPhysicalEntity>> CollidingCouples(List<IPhysicalEntity> physicalEntities)
        {
            List<Tuple<IPhysicalEntity, IPhysicalEntity>> collidingCouples = new List<Tuple<IPhysicalEntity, IPhysicalEntity>>();

            for (int i = 0; i < physicalEntities.Count - 1; i++)
            {
                IPhysicalEntity firstEntity = physicalEntities[i];
                for (int j = i + 1; j < physicalEntities.Count; j++)
                {
                    IPhysicalEntity secondEntity = physicalEntities[j];
                    if (firstEntity.CollidesWith(secondEntity))
                    {
                        collidingCouples.Add(new Tuple<IPhysicalEntity, IPhysicalEntity>(firstEntity, secondEntity));
                    }
                }
            }
            return collidingCouples;
        }
    }
}
