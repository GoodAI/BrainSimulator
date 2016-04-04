using System;
using System.Collections.Generic;
using VRageMath;
using World.GameActors.GameObjects;

namespace World.ToyWorldCore
{
    public class SimpleObjectLayer : IObjectLayer 
    {
        public List<GameObject> GameObjects { get; set; }

        public LayerType LayerType { get; set; }

        public SimpleObjectLayer(LayerType layerType)
        {
            LayerType = layerType;
            GameObjects = new List<GameObject>();
        }

        public List<GameObject> GetGameObjects(RectangleF rectangle)
        {
            throw new NotImplementedException();
        }
    }
}
