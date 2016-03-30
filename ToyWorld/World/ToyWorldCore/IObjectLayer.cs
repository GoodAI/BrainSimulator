using System.Collections.Generic;
using World.GameActors.GameObjects;

namespace World.ToyWorldCore
{
    public interface IObjectLayer
    {
        LayerType LayerType { get; set; }

        List<GameObject> GetGameObjects(VRageMath.RectangleF rectangle);
    }
}
