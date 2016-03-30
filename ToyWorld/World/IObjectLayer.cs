using System.Collections.Generic;
using VRageMath;
using World.GameActors.GameObjects;

namespace World
{
    public interface IObjectLayer
    {
        List<GameObject> GetGameObjects(VRageMath.RectangleF rectangle);
    }
}
