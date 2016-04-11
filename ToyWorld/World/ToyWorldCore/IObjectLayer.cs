using System.Collections.Generic;
using World.GameActors.GameObjects;

namespace World.ToyWorldCore
{
    public interface IObjectLayer
    {
        LayerType LayerType { get; set; }

        /// <summary>
        /// Returns objects that have intersection with giver rectangle.
        /// </summary>
        /// <param name="rectangle"></param>
        /// <returns></returns>
        List<GameObject> GetGameObjects(VRageMath.RectangleF rectangle);

        /// <summary>
        /// Adds game object to the layer. If object cannot be added, return false.
        /// </summary>
        /// <param name="gameObject">GameObject to add</param>
        /// <returns>True if object was successfully added.</returns>
        bool AddGameObject(GameObject gameObject);
    }
}
