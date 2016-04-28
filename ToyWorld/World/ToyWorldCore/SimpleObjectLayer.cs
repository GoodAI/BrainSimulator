using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using VRageMath;
using World.GameActors.GameObjects;
using World.Physics;

namespace World.ToyWorldCore
{
    public class SimpleObjectLayer : IObjectLayer
    {
        private List<GameObject> GameObjects { get; set; }

        public LayerType LayerType { get; set; }

        public SimpleObjectLayer(LayerType layerType)
        {
            LayerType = layerType;
            GameObjects = new List<GameObject>();
        }

        public GameObject GetActorAt(int x, int y)
        {
            return GetGameObjects(new RectangleF(x, y, 1, 1)).FirstOrDefault();
        }

        public List<GameObject> GetGameObjects(RectangleF rectangle)
        {
            List<GameObject> list = new List<GameObject>();

            foreach (GameObject gameObject in GameObjects)
            {
                IPhysicalEntity physicalEntity = gameObject.PhysicalEntity;
                RectangleF r = new RectangleF();
                RectangleF cover = physicalEntity.CoverRectangle();
                RectangleF.Intersect(ref cover, ref rectangle, out r);
                if (r.Size.Length() > 0f)
                {
                    list.Add(gameObject);
                }
            }

            return list;
        }

        public List<GameObject> GetGameObjects()
        {
            var newList = new List<GameObject>();
            newList.AddRange(GameObjects);
            return newList;
        }

        public List<GameObject> GetGameObjects(VRageMath.Circle circle)
        {
            var list = new List<GameObject>();

            foreach (GameObject gameObject in GameObjects)
            {
                if (circle.Include(gameObject.Position))
                {
                    list.Add(gameObject);
                }
            }

            return list;
        }

        public bool AddGameObject(GameObject gameObject)
        {
            GameObjects.Add(gameObject);
            return true;
        }

        public List<GameObject> GetAllObjects()
        {
            Contract.Ensures(Contract.Result<List<GameObject>>() != null);

            return GameObjects;
        }

        public List<IPhysicalEntity> GetPhysicalEntities(VRageMath.Circle circle)
        {
            return GetGameObjects(circle).Select(x => x.PhysicalEntity).ToList();
        }

        public List<IPhysicalEntity> GetPhysicalEntities()
        {
            return GetGameObjects().Select(x => x.PhysicalEntity).ToList();
        }

        public List<IPhysicalEntity> GetPhysicalEntities(RectangleF rectangle)
        {
            return GetGameObjects(rectangle).Select(x => x.PhysicalEntity).ToList();
        }
    }
}
