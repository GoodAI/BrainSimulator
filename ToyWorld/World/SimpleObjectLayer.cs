using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using VRageMath;
using World.GameActors.GameObjects;

namespace World
{
    public class SimpleObjectLayer : IObjectLayer 
    {
        public List<GameObject> GameObjects { get; set; }
    
        public List<GameObject> GetGameObjects(RectangleF rectangle)
        {
            throw new NotImplementedException();
        }
    }
}
