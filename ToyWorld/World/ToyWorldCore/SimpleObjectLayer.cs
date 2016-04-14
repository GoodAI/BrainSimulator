using System;
using System.Collections.Generic;
using System.Linq;
using VRageMath;
using World.GameActors.GameObjects;

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

        public List<GameObject> GetGameObjects(RectangleF rectangle)
        {
            // TODO game object bounding boxes, remove default for null positions
            return GameObjects.Where(o => rectangle.Contains(o.PhysicalEntity != null ? o.PhysicalEntity.Position : Vector2.One)).ToList();
        }

        public bool AddGameObject(GameObject gameObject)
        {
            GameObjects.Add(gameObject);
            return true;
        }

        public List<GameObject> GetAllObjects()
        {
            return GameObjects;
        }
    }
}
