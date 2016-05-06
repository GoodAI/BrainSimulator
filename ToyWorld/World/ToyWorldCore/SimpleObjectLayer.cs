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
        private List<IGameObject> GameObjects { get; set; }

        public LayerType LayerType { get; set; }

        public SimpleObjectLayer(LayerType layerType)
        {
            LayerType = layerType;
            GameObjects = new List<IGameObject>();
        }

        public List<IGameObject> GetGameObjects(RectangleF rectangle)
        {
            var list = new List<IGameObject>();

            foreach (IGameObject gameObject in GameObjects)
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

        public List<IGameObject> GetGameObjects()
        {
            var newList = new List<IGameObject>();
            newList.AddRange(GameObjects);
            return newList;
        }

        public List<IGameObject> GetGameObjects(VRageMath.Circle circle)
        {
            var list = new List<IGameObject>();

            foreach (IGameObject gameObject in GameObjects)
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

        public List<IGameObject> GetAllObjects()
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
